"""次トークン候補の探索ツール(「モデルは何と何で迷っているか」を木で見る)。

チェックポイントを読み込み、与えた出だし(トークン列)に対して上位 k 候補と確率、
その分布のエントロピー(迷い度)を返す。explore.html がこれを木として描画する。
model_inference.drawio の「次トークン確率」を 1 段ずつ目視するための道具で、
フェーズ 4 の「モデル内部可視化」の最初の一歩。

使い方:
  # 対話(ブラウザ): サーバ機で起動し、LAN の PC から http://<サーバ IP>:8765/ を開く
  python explore.py <ckpt.pt> --sp <sp_bpe_48k.model> serve [--host 0.0.0.0] [--port 8765]

  # 静的書き出し: 指定の出だしから深さ depth・各段 top 候補の木を JSON に
  python explore.py <ckpt.pt> --sp <sp_bpe_48k.model> export --prompt "The central bank" [--depth 3] [--top 5] [--out tree.json]

API(serve):
  GET /api/encode?text=...            → {"ids": [...], "pieces": [...]}
  GET /api/next?ids=1,2,3&k=8         → {"candidates": [{"id","piece","text","p"}...], "entropy": float, "top_p_mass": float}
  GET /                               → explore.html
S モデルの 1 回の推論は CPU でも数十 ms(ctx 1024 以下)。GPU があればそれを使う。
"""

import argparse
import json
import math
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sentencepiece as spm
import torch

from config import EOS_ID
from eval import load_model

HERE = Path(__file__).resolve().parent


class Explorer:
    def __init__(self, ckpt, sp_model, device):
        self.device = torch.device(device)
        self.model, self.cfg, self.step = load_model(ckpt, self.device)
        self.model.eval()
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)

    def encode(self, text):
        ids = [self.sp.bos_id()] + (self.sp.encode(text) if text else [])
        return {"ids": ids, "pieces": [self.piece(i) for i in ids]}

    def piece(self, i):
        return self.sp.id_to_piece(int(i))

    @torch.no_grad()
    def next_candidates(self, ids, k=8):
        """ids(先頭に BOS を含む)の続きとして、上位 k トークンと確率を返す。"""
        ids = ids[-self.cfg.ctx:]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(x)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        ent = float(-(probs * torch.log(probs + 1e-12)).sum())
        top_p, top_i = probs.topk(k)
        cands = []
        for p, i in zip(top_p.tolist(), top_i.tolist()):
            piece = self.piece(i)
            cands.append({"id": i, "piece": piece, "text": self.sp.decode([i]) if i != EOS_ID else "</s>", "p": p})
        return {"candidates": cands, "entropy": ent, "perplexity_here": math.exp(ent),
                "top_p_mass": float(top_p.sum())}

    def tree(self, ids, depth, top, beam):
        """静的書き出し用: 各段で上位 top を列挙し、そのうち確率上位 beam 本だけを更に掘る。"""
        node = self.next_candidates(ids, top)
        children = []
        for j, c in enumerate(node["candidates"]):
            child = {"id": c["id"], "piece": c["piece"], "text": c["text"], "p": c["p"]}
            if depth > 1 and j < beam and c["id"] != EOS_ID:
                sub = self.tree(ids + [c["id"]], depth - 1, top, beam)
                child.update({"entropy": sub["entropy"], "children": sub["children"]})
            children.append(child)
        return {"entropy": node["entropy"], "children": children}


def make_handler(ex):
    html = (HERE / "explore.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            try:
                if u.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                elif u.path == "/api/encode":
                    self._json(ex.encode(q.get("text", [""])[0]))
                elif u.path == "/api/next":
                    ids = [int(t) for t in q.get("ids", [""])[0].split(",") if t]
                    k = max(1, min(50, int(q.get("k", ["8"])[0])))
                    self._json(ex.next_candidates(ids or [ex.sp.bos_id()], k))
                elif u.path == "/api/info":
                    self._json({"step": ex.step, "config": ex.cfg.to_dict(), "device": str(ex.device)})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001 - 探索ツールなのでエラーは JSON で返して継続
                self._json({"error": str(e)}, 500)

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--sp", required=True)
    ap.add_argument("--device", default="auto")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    e = sub.add_parser("export")
    e.add_argument("--prompt", default="")
    e.add_argument("--depth", type=int, default=3)
    e.add_argument("--top", type=int, default=5, help="各段で列挙する候補数")
    e.add_argument("--beam", type=int, default=3, help="そのうち更に掘る本数")
    e.add_argument("--out", default="")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ex = Explorer(args.ckpt, args.sp, device)
    print("loaded step %s on %s" % (ex.step, device), file=sys.stderr)

    if args.cmd == "serve":
        srv = ThreadingHTTPServer((args.host, args.port), make_handler(ex))
        print("serving on http://%s:%d/" % (args.host, args.port), file=sys.stderr)
        srv.serve_forever()
    else:
        enc = ex.encode(args.prompt)
        out = {"prompt": args.prompt, "ids": enc["ids"], "pieces": enc["pieces"], "step": ex.step,
               "depth": args.depth, "top": args.top, "beam": args.beam,
               "tree": ex.tree(enc["ids"], args.depth, args.top, args.beam)}
        text = json.dumps(out, ensure_ascii=False, indent=1)
        if args.out:
            Path(args.out).write_text(text)
            print("wrote", args.out, file=sys.stderr)
        else:
            print(text)


if __name__ == "__main__":
    main()
