"""シャード(uint16 トークン列、記事間 EOS)の読み込み。

- 学習: 複数シャードからトークン数に比例して 1 シャードを選び、ランダム位置から ctx+1 個を切り出す
  (記事境界は跨いでよい。EOS が区切りとして学習される)。
- 評価: 1 シャードを先頭から ctx 幅で重ならずに走査。<name>.langs.u8(記事ごとの言語番号)があれば
  トークンごとに言語を割り当て、言語別パープレキシティを出す(review20 の目的)。
"""

import json
from pathlib import Path

import numpy as np
import torch

from config import EOS_ID, LANGS


def load_shard(path):
    """メモリマップで開く(100MB を読み込まずに済む)。"""
    return np.memmap(path, dtype=np.uint16, mode="r")


def shard_meta(path):
    p = Path(str(path) + ".json")
    return json.loads(p.read_text()) if p.exists() else {"tokens": int(Path(path).stat().st_size // 2)}


class TrainSampler:
    def __init__(self, shard_paths, ctx, seed=0):
        self.paths = [Path(p) for p in shard_paths]
        assert self.paths, "no train shards"
        self.ctx = ctx
        self.data = [load_shard(p) for p in self.paths]
        sizes = np.array([len(d) for d in self.data], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size, device):
        xs, ys = [], []
        for _ in range(batch_size):
            i = self.rng.choice(len(self.data), p=self.weights)
            d = self.data[i]
            start = self.rng.integers(0, len(d) - self.ctx - 1)
            chunk = torch.from_numpy(d[start:start + self.ctx + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.stack(xs).to(device, non_blocking=True)
        y = torch.stack(ys).to(device, non_blocking=True)
        return x, y

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, s):
        self.rng.bit_generator.state = s["rng"]


def token_langs(shard_path, tokens):
    """<shard>.langs.u8(記事順の言語番号)から、トークンごとの言語番号(uint8)を作る。
    記事 i は i 番目の EOS で終わるので、EOS の累積で記事番号が決まる。副ファイルが無ければ None。"""
    p = Path(str(shard_path).replace(".bin", ".langs.u8"))
    if not p.exists():
        return None
    doc_lang = np.fromfile(p, dtype=np.uint8)
    doc_idx = np.cumsum(np.asarray(tokens) == EOS_ID)      # EOS 自身はその記事に属する
    doc_idx = np.concatenate(([0], doc_idx[:-1]))
    n_docs_in_shard = int(doc_idx[-1]) + 1
    assert n_docs_in_shard <= len(doc_lang), "langs 副ファイルの記事数が足りません: %s" % p
    return doc_lang[doc_idx]


class EvalShard:
    def __init__(self, shard_path, ctx, max_tokens=0):
        self.path = Path(shard_path)
        self.ctx = ctx
        d = load_shard(self.path)
        n = len(d) if not max_tokens else min(len(d), max_tokens)
        n = (n // (ctx + 1)) * (ctx + 1)          # ctx+1 の窓に切り揃える
        self.tokens = np.asarray(d[:n])
        self.langs = token_langs(self.path, self.tokens)  # None なら言語別は出せない
        self.n_windows = n // (ctx + 1)

    def windows(self, batch_size, device):
        L = self.ctx + 1
        for s in range(0, self.n_windows, batch_size):
            e = min(s + batch_size, self.n_windows)
            chunk = torch.from_numpy(self.tokens[s * L:e * L].astype(np.int64)).view(e - s, L)
            lang = None
            if self.langs is not None:
                lang = torch.from_numpy(self.langs[s * L:e * L].astype(np.int64)).view(e - s, L)[:, 1:]
            yield chunk[:, :-1].to(device), chunk[:, 1:].to(device), (lang.to(device) if lang is not None else None)


@torch.no_grad()
def evaluate(model, shard_path, ctx, device, batch_size=8, max_tokens=0, autocast_dtype=None):
    """言語別 + 全体のパープレキシティ。戻り値: {"all": {"ppl", "tokens"}, "en": {...}, ...}"""
    model.eval()
    ev = EvalShard(shard_path, ctx, max_tokens)
    nll = torch.zeros(len(LANGS) + 1, dtype=torch.float64, device=device)
    cnt = torch.zeros(len(LANGS) + 1, dtype=torch.float64, device=device)
    for x, y, lang in ev.windows(batch_size, device):
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits, _ = model(x)
        l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none")
        nll[-1] += l.sum()
        cnt[-1] += l.numel()
        if lang is not None:
            lf = lang.reshape(-1)
            for li in range(len(LANGS)):
                m = lf == li
                nll[li] += l[m].sum()
                cnt[li] += m.sum()
    model.train()
    out = {"all": {"ppl": float(torch.exp(nll[-1] / cnt[-1])), "tokens": int(cnt[-1])}}
    if ev.langs is not None:
        for li, name in enumerate(LANGS):
            if cnt[li] > 0:
                out[name] = {"ppl": float(torch.exp(nll[li] / cnt[li])), "tokens": int(cnt[li])}
    return out
