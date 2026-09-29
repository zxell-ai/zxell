"""単一マシンでの学習ループ(フェーズ 1 のベースライン。フェーズ 2 ではクライアントの局所学習にも流用する)。

使い方:
  python train.py --preset S --shards <dir> --out runs/S_baseline --steps 20000
  python train.py --preset tiny --shards <dir> --out runs/smoke --steps 50 --batch 4   # CPU スモーク
  python train.py ... --resume runs/S_baseline/ckpt_last.pt                             # 再開

<dir> には train_*.bin と val_000.bin(+ .json / .langs.u8)を置く(fetch_shards.py で API から取得)。
学習: AdamW(β 0.9/0.95, wd 0.1)、線形ウォームアップ → cosine 減衰(最終 lr は 10%)、勾配クリップ 1.0、
CUDA では bf16 autocast。eval_every ステップごとに val の言語別 PPL を測り、ログ(jsonl)とチェックポイントを書く。
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from config import PRESETS, ModelConfig
from data import TrainSampler, evaluate
from model import GPT


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step >= args.steps:
        return args.lr * args.min_lr_ratio
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def save_ckpt(path, model, opt, sampler, step, cfg, args):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "sampler": sampler.state_dict(),
                "step": step, "config": cfg.to_dict(), "args": vars(args)}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="S", choices=list(PRESETS))
    ap.add_argument("--shards", required=True, help="train_*.bin / val_000.bin があるディレクトリ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=0, help="0 = preset の ctx")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-tokens", type=int, default=1_000_000, help="途中評価で見る val トークン数")
    ap.add_argument("--final-eval-tokens", type=int, default=0, help="最終評価の val トークン数(0 = 全量。CPU スモークでは小さく)")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-train-shards", type=int, default=0, help="スモーク用: 先頭 N シャードだけ使う")
    args = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shards = Path(args.shards)
    train_paths = sorted(shards.glob("train_*.bin"))
    if args.max_train_shards:
        train_paths = train_paths[:args.max_train_shards]
    val_path = shards / "val_000.bin"
    assert train_paths, "train_*.bin が見つかりません: %s" % shards

    cfg = PRESETS[args.preset]
    if args.ctx:
        cfg = ModelConfig(**{**cfg.to_dict(), "ctx": args.ctx})
    model = GPT(cfg).to(device)
    print("preset %s: %.1fM params (%.1fM non-embedding), ctx %d, device %s" % (
        args.preset, model.num_params() / 1e6, model.num_params(True) / 1e6, cfg.ctx, device), flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))
    sampler = TrainSampler(train_paths, cfg.ctx, seed=args.seed)

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sampler.load_state_dict(ck["sampler"])
        step = ck["step"]
        print("resumed from %s at step %d" % (args.resume, step), flush=True)

    (out / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args),
                                                 "train_shards": [p.name for p in train_paths]}, indent=2))
    log = open(out / "log.jsonl", "a")
    tokens_per_step = args.batch * args.grad_accum * cfg.ctx
    t0 = time.time()
    model.train()
    while step < args.steps:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = sampler.batch(args.batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            rec = {"step": step, "loss": round(loss_acc, 4), "lr": lr, "grad_norm": round(float(gn), 3),
                   "tok_per_s": round(step * tokens_per_step / dt), "elapsed_s": round(dt)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            full = step == args.steps
            ev = evaluate(model, val_path, cfg.ctx, device,
                          max_tokens=args.final_eval_tokens if full else args.eval_tokens,
                          autocast_dtype=autocast_dtype)
            rec = {"step": step, "eval": "val_full" if full else "val", **{k: round(v["ppl"], 2) for k, v in ev.items()}}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.ckpt_every == 0 or step == args.steps:
            save_ckpt(out / "ckpt_last.pt", model, opt, sampler, step, cfg, args)
    save_ckpt(out / "ckpt_final.pt", model, opt, sampler, step, cfg, args)
    print("done: %d steps, %.2fM tokens, %.0fs" % (step, step * tokens_per_step / 1e6, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
