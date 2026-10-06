"""GPT 系デコーダオンリー Transformer(自前実装・PyTorch)。

構成は _private/reports/model_architecture.drawio のとおり:
Token Embedding(LM Head と重み共有)→ [RMSNorm → causal Self-Attention(RoPE)→ 残差 →
RMSNorm → SwiGLU FFN → 残差] × N → Final RMSNorm → LM Head → 次トークン確率。
事前学習済み重みは使わない。位置は RoPE のみ(学習する位置埋め込みは持たない)。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig

# 損失計算を一度に流すシーケンス数。logits(T × 48k 語彙)が bf16 でも 1 本 98MB・fp32 で 197MB あり、
# batch 32 を一括にすると 6GB 超(RTX 3060 で OOM — 2026-10-06)。LM Head + 損失をこの単位で分割し、
# 逆伝播時に再計算(checkpoint)することで、保持する logits をチャンク 1 個分に抑える。
LOSS_CHUNK_SEQS = 4


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # fp32 で分散を計算してから元の dtype に戻す(bf16 学習時の安定性)
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def rope_cache(head_dim, ctx, theta, device):
    """RoPE の cos / sin テーブル(ctx, head_dim/2)。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(ctx, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, D)。D を前半・後半に分けて回転させる(GPT-NeoX 方式)。"""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal マスク込み
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.ffn_hidden
        self.w_gate = nn.Linear(cfg.d_model, h, bias=False)
        self.w_up = nn.Linear(cfg.d_model, h, bias=False)
        self.w_down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # 残差に足し込む射影は層数に応じて小さく初期化する(GPT-2 流)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self._rope = {}

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def _rope_for(self, device):
        key = str(device)
        if key not in self._rope:
            self._rope[key] = rope_cache(self.cfg.head_dim, self.cfg.ctx, self.cfg.rope_theta, device)
        return self._rope[key]

    def forward(self, idx, targets=None):
        """idx: (B, T) トークン ID。targets があれば次トークン予測の平均 loss も返す。"""
        B, T = idx.shape
        assert T <= self.cfg.ctx, "sequence length %d > ctx %d" % (T, self.cfg.ctx)
        cos, sin = self._rope_for(idx.device)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        if targets is None:
            return self.lm_head(x), None
        # 学習時: logits 全体は返さず(メモリ節約)、チャンクごとに LM Head → 損失を計算する
        total = x.new_zeros((), dtype=torch.float32)
        for xs, ts in zip(x.split(LOSS_CHUNK_SEQS, dim=0), targets.split(LOSS_CHUNK_SEQS, dim=0)):
            total = total + checkpoint(self._chunk_loss, xs, ts, use_reentrant=False)
        return None, total / targets.numel()

    def _chunk_loss(self, x, targets):
        logits = self.lm_head(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="sum")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=1.0, eos_id=None):
        """自己回帰生成(model_inference.drawio)。KV キャッシュなしの素朴版(デモ・評価用)。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    keep = (sp.cumsum(-1) - sp) < top_p
                    sp = sp * keep
                    probs = torch.zeros_like(probs).scatter_(-1, si, sp)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
