"""チェックポイントの言語別パープレキシティ評価(val / test)。

使い方: python eval.py <ckpt.pt> <shards_dir> [val|test] [--max-tokens N]
出力: JSON(全体と言語別の PPL・評価トークン数)。フェーズ 1 の完了判定と、フェーズ 2 以降の
「単一マシン学習比で劣化 10% 以内」(成功基準 1.2)の比較に使う。
"""

import argparse
import json
from pathlib import Path

import torch

from config import ModelConfig
from data import evaluate
from model import GPT


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ck["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg, ck.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("shards")
    ap.add_argument("split", nargs="?", default="val", choices=["val", "test"])
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, cfg, step = load_model(args.ckpt, device)
    shard = Path(args.shards) / ("%s_000.bin" % args.split)
    res = evaluate(model, shard, cfg.ctx, device, batch_size=args.batch, max_tokens=args.max_tokens,
                   autocast_dtype=torch.bfloat16 if device.type == "cuda" else None)
    print(json.dumps({"ckpt": args.ckpt, "step": step, "split": args.split, "ctx": cfg.ctx, "result": res},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
