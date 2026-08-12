"""フェーズ1: トークナイザ学習・評価用のサンプルコーパス抽出。

sphered_production.feed_items を 1 回だけストリーミングで全走査し、
- 全言語: id % 20 == 0 の 5% サンプル
- 日本語: id % 4 == 0 に増量（文字数が全体の 1.5% しかないため、語彙学習用に厚めに取る）
を言語別ファイルに書き出す。読み取りは SELECT のみ。

テキスト整形は本番前処理と同じ方針（roadmap 5 章 / phase0 の合意）:
- 空白・改行・タブは 1 個のスペースに正規化（1 記事 1 行で出力するため）
- ja のみ title + description + content_text を連結、他言語は content_text のみ
- サンプル内で link 正規化キーによる重複除去

出力: {outdir}/train_{lang}.txt, {outdir}/eval_{lang}.txt（50 記事に 1 件、言語あたり
最大 3,000 記事を評価用に確保）, {outdir}/stats.json

使い方: python extract_sample.py /mnt/exssd/zxell/work/phase1
接続情報は ~/.my.cnf（[client] セクション）を参照する。パスワードは扱わない。
"""

import json
import re
import sys
import time
from pathlib import Path

import pymysql

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

    conn = pymysql.connect(
        read_default_file="~/.my.cnf",
        unix_socket="/var/run/mysqld/mysqld.sock",  # TCP は無効化されているためソケット接続
        database="sphered_production",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.SSCursor,  # ストリーミング（全件をメモリに載せない）
    )
    files = {}
    for lang in LANGS:
        files[("train", lang)] = open(outdir / ("train_%s.txt" % lang), "w", encoding="utf-8")
        files[("eval", lang)] = open(outdir / ("eval_%s.txt" % lang), "w", encoding="utf-8")

    stats = {lang: {"train_docs": 0, "train_chars": 0, "eval_docs": 0, "eval_chars": 0} for lang in LANGS}
    seen_links = set()
    n_rows = n_dup = n_empty = 0
    started = time.time()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.id, LEFT(c.language, 2), i.link, i.title, i.description, i.content_text
            FROM feed_items i
            JOIN feed_channels c ON c.id = i.channel_id
            WHERE MOD(i.id, 20) = 0 OR (LEFT(c.language, 2) = 'ja' AND MOD(i.id, 4) = 0)
            """
        )
        for row_id, lang, link, title, description, content in cur:
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
    conn.close()

    stats["_meta"] = {
        "rows_scanned_matched": n_rows,
        "dup_links_skipped": n_dup,
        "empty_skipped": n_empty,
        "elapsed_sec": round(time.time() - started),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
