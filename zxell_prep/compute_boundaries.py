"""フェーズ1: train / val / test の時系列分割境界を言語別に算出する(boundaries.json を出力)。

背景(review20): en/de/fr の収集は 2023-05-28 に同日で止まり、ja だけ 2024-02-12 まで続いた。
全言語共通の日付で末尾を切ると val/test が ja 100% になるため、**言語ごとに** pub_date の
新しい側から test = TEST_DOCS 記事、その直前 val = VAL_DOCS 記事を切り出す(案 A')。
「新しい記事で評価する」意図(roadmap 5 章)を言語別に保ちつつ、4 言語の val/test サイズを揃える。

境界は「その言語で新しい方から N 番目の記事の pub_date」で決める。同じ日時の記事が
境界にまたがる場合は少し多めに val/test に入る(数件〜数十件。問題にならない)。
pub_date が不正('0000-00-00' 等)・未来日付の記事は集計対象外で、tokenize_full.py 側で train に落ちる。

出力 boundaries.json の形式(tokenize_full.py が読む):
{
  "mode": "per_lang_count", "test_docs": 8000, "val_docs": 3000,
  "langs": {"en": {"max_pub_date": ..., "test_start": ..., "val_start": ..., "docs": N}, ...}
}

使い方: python compute_boundaries.py <workdir> [snapshot_dir]
snapshot_dir 省略時は環境変数 ZXELL_SNAPSHOT。
"""

import gzip
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_SNAPSHOT = "/mnt/exssd/zxell/backup/feed_items/snapshot"
LANGS = ("en", "de", "fr", "ja")
TEST_DOCS = 8000   # 言語ごとの test 記事数(最新側)
VAL_DOCS = 3000    # その直前の val 記事数


def main():
    work = Path(sys.argv[1])
    snapshot = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("ZXELL_SNAPSHOT", DEFAULT_SNAPSHOT))
    work.mkdir(parents=True, exist_ok=True)
    channels = json.loads((snapshot / "channels.json").read_text())
    now = datetime.now()
    started = time.time()

    dates = {lang: [] for lang in LANGS}
    n_rows = n_bad = n_future = 0
    with gzip.open(snapshot / "feed_items.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lang = (channels.get(row[1]) or "??")[:2]
            if lang not in dates:
                continue
            n_rows += 1
            try:
                d = datetime.fromisoformat(row[6])
            except (ValueError, TypeError):
                n_bad += 1
                continue
            if d > now:
                n_future += 1
                continue
            dates[lang].append(d)

    out = {"mode": "per_lang_count", "test_docs": TEST_DOCS, "val_docs": VAL_DOCS,
           "computed_at": now.isoformat(timespec="seconds"), "snapshot": str(snapshot), "langs": {}}
    for lang in LANGS:
        ds = sorted(dates[lang], reverse=True)          # 新しい順
        if len(ds) < TEST_DOCS + VAL_DOCS + 1:
            sys.exit("%s: 記事数 %d が不足(test+val=%d 必要)" % (lang, len(ds), TEST_DOCS + VAL_DOCS))
        test_start = ds[TEST_DOCS - 1]                  # 新しい方から TEST_DOCS 番目
        val_start = ds[TEST_DOCS + VAL_DOCS - 1]        # その直前 VAL_DOCS 件
        out["langs"][lang] = {
            "docs": len(ds),
            "max_pub_date": ds[0].isoformat(),
            "test_start": test_start.isoformat(),
            "val_start": val_start.isoformat(),
            # 同時刻の記事が境界にまたがった分を含めた実際の件数
            "test_docs_actual": sum(1 for d in ds if d >= test_start),
            "val_docs_actual": sum(1 for d in ds if val_start <= d < test_start),
        }
    out["_meta"] = {"rows_scanned": n_rows, "bad_pub_date": n_bad, "future_skipped": n_future,
                    "elapsed_sec": round(time.time() - started)}
    (work / "boundaries.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
