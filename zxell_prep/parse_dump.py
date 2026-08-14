"""バックアップ SQL ダンプから feed_items / feed_channels を抽出して JSONL スナップショットを作る。

2026-08-13 の InnoDB ページ破損(review14)を受けた対応。稼働中の MySQL には一切
書き込まず(復元もせず)、mysqldump ファイルを直接ストリーミングパースして、
前処理が必要とする列だけの再利用可能なスナップショットを別ディスクに作る。

入力: mysqldump 5.7 形式・列名付き extended INSERT・データのみのダンプ(.sql.gz)
出力(outdir):
  channels.json        feed_channels の id → language
  feed_items.jsonl.gz  1 行 1 記事の JSON 配列
                       [id, channel_id, link, title, description, content_text, pub_date]
  feed_items.jsonl.gz.sha256  書き込み検証に合格した時点のハッシュ
  parse_stats.json     行数・id 範囲・言語別件数など

書き終えた gz は必ず読み直して検証する(2026-08-13: 本機のサイレント破損で、
書き込み後に中身が化けたスナップショットができた。gzip の CRC は圧縮器に渡した
データに対して計算されるため、読み直せば「書いた後に化けた」を検出できる)。
検証に落ちたら異常終了し、壊れたスナップショットを後段に渡さない。

使い方: python parse_dump.py <dump.sql.gz> <outdir> [max_items]
max_items を付けるとスモークテスト(feed_items を先頭 N 行で打ち切り)。
"""

import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ITEM_COLS = ("id", "channel_id", "link", "title", "description", "content_text", "pub_date")

STR = re.compile(r"'(?:[^'\\]|\\.)*'")
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
INSERT = re.compile(r"^INSERT INTO `(\w+)` \(([^)]*)\) VALUES ")
# 対象 2 テーブル以外の行はデコードせずに素通しする(他テーブルに非 UTF-8 バイトが
# 混在しているため。対象行は replace デコードし、置換発生数を stats に記録する)
INSERT_B = re.compile(rb"^INSERT INTO `(feed_channels|feed_items)` ")
ESC = {"0": "\0", "'": "'", '"': '"', "b": "\b", "n": "\n", "r": "\r",
       "t": "\t", "Z": "\x1a", "\\": "\\", "%": "%", "_": "_"}
UNESC = re.compile(r"\\(.)")


def unescape(s):
    if "\\" not in s:
        return s
    return UNESC.sub(lambda m: ESC.get(m.group(1), m.group(1)), s)


def parse_tuples(s, pos, ncols):
    """VALUES 以降を走査し、値タプルを順に yield する(文字列内の '),(' に騙されない)。"""
    while True:
        if s[pos] != "(":
            raise ValueError("expected '(' at %d: %r" % (pos, s[pos:pos + 40]))
        pos += 1
        vals = []
        for ci in range(ncols):
            c = s[pos]
            if c == "'":
                m = STR.match(s, pos)
                vals.append(unescape(m.group(0)[1:-1]))
                pos = m.end()
            elif s.startswith("NULL", pos):
                vals.append(None)
                pos += 4
            else:
                m = NUM.match(s, pos)
                if not m:
                    raise ValueError("bad value at %d: %r" % (pos, s[pos:pos + 40]))
                vals.append(m.group(0))
                pos = m.end()
            if ci < ncols - 1:
                if s[pos] != ",":
                    raise ValueError("expected ',' at %d: %r" % (pos, s[pos:pos + 40]))
                pos += 1
        if s[pos] != ")":
            raise ValueError("expected ')' at %d: %r" % (pos, s[pos:pos + 40]))
        pos += 1
        yield vals
        if pos < len(s) and s[pos] == ",":
            pos += 1
            continue
        return


def verify_output(path, expected_items):
    """書き終えた gz を読み直し、gzip CRC・行数・全行の JSON 妥当性を検証する。

    gzip の CRC は圧縮器に渡した非圧縮データに対して計算されて末尾に記録されるので、
    読み直して CRC が合わなければ「書いた後に化けた」ことがわかる(RAM か書き込み経路)。
    合格した場合だけ sha256 を書き出し、それを後段の正とする。
    """
    print("検証: %s を読み直します(数分かかります)" % path.name, flush=True)
    lines = bad_json = 0
    started = time.time()
    h = hashlib.sha256()
    with open(path, "rb") as raw:
        # 直前に書いたばかりでページキャッシュに載っているので、実ディスクから
        # 読み直すために書き戻してから捨てる(捨てられない環境でも検証自体は成立する)
        try:
            os.fsync(raw.fileno())
            os.posix_fadvise(raw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        for chunk in iter(lambda: raw.read(1 << 20), b""):
            h.update(chunk)
    try:
        with gzip.open(path, "rb") as f:
            for bline in f:
                lines += 1
                try:
                    row = json.loads(bline.decode("utf-8"))
                    if not (isinstance(row, list) and len(row) == len(ITEM_COLS)):
                        raise ValueError("列数が違う")
                except (UnicodeDecodeError, ValueError) as e:
                    bad_json += 1
                    if bad_json <= 5:
                        print("  壊れた行 %d: %r (%s)" % (lines, bline[:120], e), flush=True)
    except gzip.BadGzipFile as e:
        raise SystemExit("検証失敗: gzip CRC 不一致 = 書き込み後にデータが化けています (%s)\n"
                         "  → RAM または書き込み経路の異常です。memtest86+ と書き込み先の変更を検討してください。" % e)
    if lines != expected_items or bad_json:
        raise SystemExit("検証失敗: 書いた行数 %d に対し読めた行数 %d、壊れた行 %d"
                         % (expected_items, lines, bad_json))
    (path.parent / (path.name + ".sha256")).write_text("%s  %s\n" % (h.hexdigest(), path.name))
    print("検証 OK: %d 行・CRC 一致・sha256 %s (%.0f 秒)"
          % (lines, h.hexdigest()[:16], time.time() - started), flush=True)


def main():
    dump_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    max_items = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    outdir.mkdir(parents=True, exist_ok=True)

    channels = {}
    stats = {"items": 0, "min_id": None, "max_id": None, "null_content": 0,
             "lang_docs": {}, "insert_statements": 0, "replaced_chars": 0}
    out = gzip.open(outdir / "feed_items.jsonl.gz", "wt", encoding="utf-8", compresslevel=5)
    started = time.time()
    done = False

    with gzip.open(dump_path, "rb") as f:
        for bline in f:
            if not INSERT_B.match(bline):
                continue
            line = bline.decode("utf-8", errors="replace")
            m = INSERT.match(line)
            if not m:
                continue
            table = m.group(1)
            if table == "feed_items":
                stats["replaced_chars"] += line.count("�")
            cols = [c.strip().strip("`") for c in m.group(2).split(",")]
            stats["insert_statements"] += 1
            if table == "feed_channels":
                i_id, i_lang = cols.index("id"), cols.index("language")
                for vals in parse_tuples(line.rstrip("\n").rstrip(";"), m.end(), len(cols)):
                    channels[vals[i_id]] = vals[i_lang]
            else:
                idx = [cols.index(c) for c in ITEM_COLS]
                for vals in parse_tuples(line.rstrip("\n").rstrip(";"), m.end(), len(cols)):
                    row = [vals[i] for i in idx]
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["items"] += 1
                    rid = int(row[0])
                    stats["min_id"] = rid if stats["min_id"] is None else min(stats["min_id"], rid)
                    stats["max_id"] = rid if stats["max_id"] is None else max(stats["max_id"], rid)
                    if row[5] is None:
                        stats["null_content"] += 1
                    lang = (channels.get(row[1]) or "??")[:2]
                    stats["lang_docs"][lang] = stats["lang_docs"].get(lang, 0) + 1
                    if stats["items"] % 500000 == 0:
                        print("... %d items (%.0f sec)" % (stats["items"], time.time() - started), flush=True)
                    if max_items and stats["items"] >= max_items:
                        done = True
                        break
            if done:
                break
    out.close()
    (outdir / "channels.json").write_text(json.dumps(channels, indent=0, ensure_ascii=False))
    stats["channels"] = len(channels)
    stats["max_items_limit"] = max_items
    stats["elapsed_sec"] = round(time.time() - started)
    (outdir / "parse_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "lang_docs"}, indent=2))
    print("lang_docs:", json.dumps(stats["lang_docs"], ensure_ascii=False))
    verify_output(outdir / "feed_items.jsonl.gz", stats["items"])


if __name__ == "__main__":
    main()
