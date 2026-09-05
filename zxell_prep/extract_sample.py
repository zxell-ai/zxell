"""フェーズ1: トークナイザ学習・評価用のサンプルコーパス抽出。

parse_dump.py が作るバックアップスナップショット(feed_items.jsonl.gz + channels.json)を
1 回だけストリーミングで全走査し、
- 全言語: id % 20 == 0 の 5% サンプル
- 日本語: id % 4 == 0 に増量（文字数が全体の 1.5% しかないため、語彙学習用に厚めに取る）
を言語別ファイルに書き出す。

〔2026-09-05 改訂〕旧版は稼働 MySQL を読んでいたが、旧サーバの /ssd 故障で
稼働 DB ごと失われたため、スナップショット読みに変更した(コーパスの正は
2025-03-30 のバックアップ dump → parse_dump.py のスナップショット)。
抽出条件・整形・重複除去の方針は旧版と同一。

テキスト整形は本番前処理と同じ方針（roadmap 5 章 / phase0 の合意）:
- 空白・改行・タブは 1 個のスペースに正規化（1 記事 1 行で出力するため）
- ja のみ title + description + content_text を連結、他言語は content_text のみ
- サンプル内で link 正規化キーによる重複除去

出力: {outdir}/train_{lang}.txt, {outdir}/eval_{lang}.txt（50 記事に 1 件、言語あたり
最大 3,000 記事を評価用に確保）, {outdir}/stats.json

使い方: python extract_sample.py <outdir> [snapshot_dir]
snapshot_dir 省略時は環境変数 ZXELL_SNAPSHOT、それも無ければ
/mnt/exssd/zxell/backup/feed_items/snapshot を読む。
"""

import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_SNAPSHOT = "/mnt/exssd/zxell/backup/feed_items/snapshot"
EVAL_EVERY = 50          # サンプル行のうち 50 記事に 1 件を評価用へ
EVAL_MAX_PER_LANG = 3000
LANGS = ("en", "de", "fr", "ja")

WS = re.compile(r"\s+")


def clean(s):
    if not s:
        return ""
    return WS.sub(" ", s).strip()


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    outdir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(sys.argv[2] if len(sys.argv) > 2 else
                    os.environ.get("ZXELL_SNAPSHOT", DEFAULT_SNAPSHOT))

    # feed_channels.language の先頭 2 文字 = 言語コード(旧版の LEFT(c.language, 2) と同一)
    channels = json.loads((snapshot / "channels.json").read_text())
    lang_of = {cid: lang[:2] for cid, lang in channels.items()}

    files = {}
    for lang in LANGS:
        files[("train", lang)] = open(outdir / ("train_%s.txt" % lang), "w", encoding="utf-8")
        files[("eval", lang)] = open(outdir / ("eval_%s.txt" % lang), "w", encoding="utf-8")

    stats = {lang: {"train_docs": 0, "train_chars": 0, "eval_docs": 0, "eval_chars": 0} for lang in LANGS}
    seen_links = set()
    n_rows = n_dup = n_empty = 0
    started = time.time()

    with gzip.open(snapshot / "feed_items.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row_id_s, _channel_id, link, title, description, content, _pub_date = json.loads(line)
            row_id = int(row_id_s)
            lang = lang_of.get(_channel_id, "")
            # 旧版の WHERE MOD(i.id, 20) = 0 OR (ja AND MOD(i.id, 4) = 0) と同一
            if not (row_id % 20 == 0 or (lang == "ja" and row_id % 4 == 0)):
                continue
            n_rows += 1
            if n_rows % 100000 == 0:
                print("... %d rows (%.0f sec)" % (n_rows, time.time() - started), flush=True)
            if lang not in stats:
                continue
            key = clean(link)
            if key in seen_links:
                n_dup += 1
                continue
            seen_links.add(key)

            if lang == "ja":
                text = " ".join(t for t in (clean(title), clean(description), clean(content)) if t)
            else:
                text = clean(content)
            if len(text) < 20:
                n_empty += 1
                continue

            split = "eval" if (n_rows % EVAL_EVERY == 0 and stats[lang]["eval_docs"] < EVAL_MAX_PER_LANG) else "train"
            files[(split, lang)].write(text + "\n")
            stats[lang][split + "_docs"] += 1
            stats[lang][split + "_chars"] += len(text)

    for f in files.values():
        f.close()

    stats["_meta"] = {
        "snapshot": str(snapshot),
        "rows_scanned_matched": n_rows,
        "dup_links_skipped": n_dup,
        "empty_skipped": n_empty,
        "elapsed_sec": round(time.time() - started),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
