"""チェックポイントから文章を生成して目で見る(フェーズ 1 の判断ポイント「S がまともな文を生成するか」)。

使い方: python sample.py <ckpt.pt> --sp <sp_bpe_48k.model> [--prompt "..."] [--tokens 100] [--temp 0.8] [--top-p 0.95]
プロンプト無しなら BOS(<s>)だけから生成する。
"""

import argparse

import sentencepiece as spm
import torch

from config import EOS_ID
from eval import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--sp", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, cfg, step = load_model(args.ckpt, device)
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=args.sp)
    ids = [sp.bos_id()] + (sp.encode(args.prompt) if args.prompt else [])
    for i in range(args.n):
        torch.manual_seed(i)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, args.tokens, temperature=args.temp, top_p=args.top_p, eos_id=EOS_ID)
        print("--- sample %d (step %s) ---" % (i, step))
        print(sp.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
