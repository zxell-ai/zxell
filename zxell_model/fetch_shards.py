"""コーディネータ API からシャードを取得する(フェーズ 1 でも配布経路は API を通す — review22 4 章)。

GET /api/shards/{name} は承認済みクライアントの X-API-Key が必要。まず shards_index.json を取り、
各シャードの .bin と .json(と val/test の .langs.u8)を落として sha256 を照合する。既に正しいものは飛ばす。

使い方:
  export ZXELL_API_URL=http://192.168.1.2      # LAN 内は nginx(:80)へ直結(Cloudflare の 100MB 制限を避ける)
  export ZXELL_API_HOST=api.zxell.ai           # nginx の vhost 振り分け用 Host ヘッダ(IP 直打ち時に必要)
  export ZXELL_API_KEY=<クライアントの API キー>
  (外から使うときは ZXELL_API_URL=https://api.zxell.ai、ZXELL_API_HOST は不要)
  python fetch_shards.py <保存先dir> [--only val,test] [--train-limit N]
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


API_HOST = os.environ.get("ZXELL_API_HOST", "")


def get(url, key, dst=None):
    headers = {"X-API-Key": key}
    if API_HOST:
        headers["Host"] = API_HOST   # uvicorn は 127.0.0.1 のみ bind。LAN からは nginx 経由で vhost 名が要る
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        if dst is None:
            return r.read()
        h = hashlib.sha256()
        with open(dst, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                h.update(chunk)
        return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--only", default="", help="例: val,test / train")
    ap.add_argument("--train-limit", type=int, default=0)
    args = ap.parse_args()
    base = os.environ.get("ZXELL_API_URL", "").rstrip("/")
    key = os.environ.get("ZXELL_API_KEY", "")
    if not base or not key:
        sys.exit("ZXELL_API_URL / ZXELL_API_KEY を設定してください")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    index = json.loads(get(base + "/api/shards/shards_index.json", key))
    (dest / "shards_index.json").write_text(json.dumps(index, indent=2))
    only = set(args.only.split(",")) if args.only else None
    n_train = 0
    for m in index["shards"]:
        if only and m["split"] not in only:
            continue
        if m["split"] == "train":
            n_train += 1
            if args.train_limit and n_train > args.train_limit:
                continue
        name = m["name"]
        binp = dest / name
        if binp.exists() and sha256_file(binp) == m["sha256"]:
            print("skip (ok):", name, flush=True)
        else:
            got = get(base + "/api/shards/" + name, key, binp)
            if got != m["sha256"]:
                binp.unlink()
                sys.exit("sha256 不一致: %s(転送中の破損)。再実行してください" % name)
            print("fetched:", name, "%.1fMB" % (binp.stat().st_size / 1e6), flush=True)
        (dest / (name + ".json")).write_bytes(get(base + "/api/shards/" + name + ".json", key))
        if m["split"] in ("val", "test"):
            langs = name.replace(".bin", ".langs.u8")
            try:
                (dest / langs).write_bytes(get(base + "/api/shards/" + langs, key))
            except Exception as e:  # noqa: BLE001 - 副ファイルは無くても学習は可能(言語別 PPL が出ないだけ)
                print("warning: %s を取得できません (%s)" % (langs, e), flush=True)
    print("done")


if __name__ == "__main__":
    main()
