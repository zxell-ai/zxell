"""シャードの中身を原文にデコードして目視確認する小道具。

シャード(.bin)は uint16 のトークン ID 列で、記事間は EOS で区切られている
(tokenize_full.py の出力形式)。これを記事単位で切り出してデコード表示する。
学習データに「何がどう入っているか」を人間の目で抜き打ち確認するためのもの。

使い方:
  python peek_shard.py <シャード名|パス> [開始記事番号] [記事数]

例:
  python peek_shard.py train_000            # 先頭 3 記事を表示
  python peek_shard.py train_042 100 5      # 100 番目から 5 記事
  python peek_shard.py /path/to/val_000.bin

シャード名だけを渡すと ZXELL_SHARDS_DIR(既定: /mnt/exssd/zxell/storage/shards)から
探す。トークナイザは ZXELL_SP_MODEL(既定は tokenize_full.py と同じ)。
メタ JSON(<shard>.json)があれば言語構成などの要約も添える。
"""

import json
import os
import sys
from array import array
from pathlib import Path

import sentencepiece as spm

SHARDS_DIR = Path(os.environ.get("ZXELL_SHARDS_DIR", "/mnt/exssd/zxell/storage/shards"))
SP_MODEL = Path(os.environ.get("ZXELL_SP_MODEL", "/mnt/exssd/zxell/work/phase1/sp_bpe_48k.model"))

PREVIEW_CHARS = 400  # 1 記事あたりの表示文字数(それ以降は「…」で省略)


def resolve_shard(arg):
    p = Path(arg)
    if p.exists():
        return p
    for cand in (SHARDS_DIR / arg, SHARDS_DIR / (arg + ".bin")):
        if cand.exists():
            return cand
    sys.exit("シャードが見つかりません: %s (ZXELL_SHARDS_DIR=%s)" % (arg, SHARDS_DIR))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    shard = resolve_shard(sys.argv[1])
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    eos = sp.eos_id()

    ids = array("H")
    ids.frombytes(shard.read_bytes())

    meta_path = shard.with_suffix(".bin.json") if shard.suffix != ".json" else shard
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        langs = ", ".join("%s %.1f%%" % (k, 100 * v / meta["tokens"])
                          for k, v in sorted(meta["langs"].items(), key=lambda x: -x[1]))
        print("%s: %s トークン / %s 記事 / %s" % (shard.name, f"{meta['tokens']:,}", f"{meta['docs']:,}", langs))
    else:
        print("%s: %s トークン(メタ JSON なし)" % (shard.name, f"{len(ids):,}"))
    print("-" * 72)

    # EOS 区切りで記事単位に走査(全記事のリスト化はせず、必要な範囲だけデコード)
    doc_i = 0
    doc_start = 0
    shown = 0
    for pos, tid in enumerate(ids):
        if tid != eos:
            continue
        if doc_i >= start and shown < count:
            doc_ids = list(ids[doc_start:pos])
            text = sp.decode(doc_ids)
            head = text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")
            print("[記事 %d] %d トークン / %d 文字" % (doc_i, len(doc_ids), len(text)))
            print(head)
            print("-" * 72)
            shown += 1
        doc_i += 1
        doc_start = pos + 1
        if shown >= count:
            break

    if shown < count:
        print("(シャード末尾に到達。表示できたのは %d 記事)" % shown)


if __name__ == "__main__":
    main()
