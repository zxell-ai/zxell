"""フェーズ1: 全量トークン化+シャード化(トークナイザ 48k 採用決定後の本番前処理)。

記事全量をストリーミングで走査し、extract_sample.py と同一方針で
正規化・link 重複除去・ja 連結 → sp_bpe_48k でエンコード → 記事間に EOS を挟んで
連結 → pub_date で train/val/test を時系列分離 → 50M トークン/シャードの
uint16 生配列(.bin)+ メタ JSON(トークン数・言語構成・sha256)として出力する。

読み取り元は parse_dump.py が作るバックアップスナップショット(SNAPSHOT)。
2026-08-13 に稼働 DB の feed_items で InnoDB ページ破損が発生したため(review14)、
稼働 DB は読まない。バックアップ(2025-03-30、収集停止後なので実質全量)が正。

分割境界: MAX(pub_date)(未来日付の混入に備えて現在時刻で上限)から遡って
3 か月を test、その前 1 か月を val、残りを train(roadmap 5 章の時系列分離)。
train のみ記事単位でシャッフルしてからパックする(val/test は走査順のまま)。

使い方:
  python tokenize_full.py <workdir> encode [limit]  # スナップショット走査+エンコード(中間 .u16)
  python tokenize_full.py <workdir> shard           # シャッフル+シャード化+メタ出力
  python tokenize_full.py <workdir> all [limit]
limit(行数)を付けるとスモークテスト(先頭 N 行で打ち切り)。
シャード出力先は SHARDS_DIR。分割境界は workdir/boundaries.json(必須。稼働 DB の
pub_date 実測から 2026-08-13 に確定済みのものを使う)。
"""

import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from array import array
from datetime import datetime
from pathlib import Path

import sentencepiece as spm

# 2026-09-05: 新サーバ移行(旧 /ssd 喪失)に伴い、環境変数で場所を差し替え可能にした
SNAPSHOT = Path(os.environ.get("ZXELL_SNAPSHOT", "/mnt/exssd/zxell/backup/feed_items/snapshot"))
SP_MODEL = Path(os.environ.get("ZXELL_SP_MODEL", "/mnt/exssd/zxell/work/phase1/sp_bpe_48k.model"))
SHARDS_DIR = Path(os.environ.get("ZXELL_SHARDS_DIR", "/mnt/exssd/zxell/storage/shards"))
SHARD_TOKENS = 50_000_000        # 1 シャード 50M トークン(uint16 で約 100MB)
LANGS = ("en", "de", "fr", "ja")
SPLITS = ("train", "val", "test")
BATCH_DOCS = 512
SEED = 20260813

WS = re.compile(r"\s+")


def drop_cache(f):
    """このファイルをページキャッシュから落とす(読み直し検証を実ディスクに当てるため)。"""
    try:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def clean(s):
    if not s:
        return ""
    return WS.sub(" ", s).strip()


def link_key(link):
    # 全量では link 文字列そのものを保持せず 64bit ハッシュで重複判定する
    # (757 万件で衝突期待値 ~1e-6 件。メモリを数 GB → 数百 MB に抑える)
    return int.from_bytes(hashlib.blake2b(clean(link).encode("utf-8"), digest_size=8).digest(), "big")


def load_boundaries(work):
    b = json.loads((work / "boundaries.json").read_text())
    return datetime.fromisoformat(b["val_start"]), datetime.fromisoformat(b["test_start"])


def iter_snapshot(limit=0, damage=None):
    """スナップショットを (lang, link, title, description, content, pub_date) で流す。

    2026-08-13 のサイレント破損(review16)を受けて、壊れた行に当たっても走査を止めず、
    件数だけ damage に記録して読み飛ばす(1 行の破損で 4 時間の処理が落ちるのを防ぐ)。
    スナップショット自体の健全性は parse_dump.py の書き込み検証で担保する前提で、
    ここは最後の砦。damage が 0 でない実行結果は採用しないこと。
    """
    channels = json.loads((SNAPSHOT / "channels.json").read_text())
    if damage is None:
        damage = {}
    damage.setdefault("broken_lines", 0)
    damage.setdefault("gzip_crc_error", False)
    n = 0
    try:
        with gzip.open(SNAPSHOT / "feed_items.jsonl.gz", "rb") as f:
            for bline in f:
                try:
                    _id, channel_id, link, title, description, content, pub_date = json.loads(
                        bline.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    damage["broken_lines"] += 1
                    continue
                lang = (channels.get(channel_id) or "??")[:2]
                try:
                    pd = datetime.fromisoformat(pub_date)
                except (ValueError, TypeError):  # '0000-00-00 00:00:00' 等は train 側へ
                    pd = datetime(1970, 1, 1)
                yield lang, link, title, description, content, pd
                n += 1
                if limit and n >= limit:
                    return
    except gzip.BadGzipFile:
        # 末尾の CRC 照合失敗 = 書き込み後にファイルが化けている(review16)
        damage["gzip_crc_error"] = True


def encode(work, limit=0):
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    assert sp.vocab_size() <= 65536, "uint16 に収まらない語彙サイズ"
    eos = sp.eos_id()
    assert eos >= 0
    val_start, test_start = load_boundaries(work)

    tok_files = {s: open(work / ("tokens_%s.u16" % s), "wb") for s in SPLITS}
    lens = {s: array("I") for s in SPLITS}
    langcodes = {s: array("B") for s in SPLITS}
    stats = {s: {l: {"docs": 0, "tokens": 0, "chars": 0} for l in LANGS} for s in SPLITS}
    seen = set()
    n_rows = n_dup = n_empty = n_otherlang = 0
    n_retried = n_anomaly_skipped = 0
    started = time.time()

    def flush(batch):
        nonlocal n_retried, n_anomaly_skipped
        texts = [t for _, _, t in batch]
        try:
            encoded = sp.encode(texts, out_type=int, num_threads=2)
        except TypeError:
            encoded = sp.encode(texts, out_type=int)
        for (split, lang, text), ids in zip(batch, encoded):
            ids.append(eos)
            try:
                arr = array("H", ids)
            except (OverflowError, TypeError, ValueError):
                # 2026-08-13 の全量実行で語彙外の巨大 int が 1 度だけ混入(再現せず。
                # 本機のサイレント破損歴と同種の一過性異常とみて、単発再エンコードで回復する)
                n_retried += 1
                print("WARN: invalid ids, re-encoding solo (%r...)" % text[:80], flush=True)
                ids = sp.encode(text, out_type=int)
                ids.append(eos)
                try:
                    arr = array("H", ids)
                except (OverflowError, TypeError, ValueError):
                    n_anomaly_skipped += 1
                    print("WARN: still invalid, skipping doc (%r...)" % text[:80], flush=True)
                    continue
            arr.tofile(tok_files[split])
            lens[split].append(len(ids))
            langcodes[split].append(LANGS.index(lang))
            st = stats[split][lang]
            st["docs"] += 1
            st["tokens"] += len(ids)
            st["chars"] += len(text)

    batch = []
    damage = {}
    for lang, link, title, description, content, pub_date in iter_snapshot(limit, damage):
        n_rows += 1
        if n_rows % 200000 == 0:
            done = sum(st["tokens"] for s in SPLITS for st in stats[s].values())
            print("... %d rows, %.1fM tokens (%.0f sec)"
                  % (n_rows, done / 1e6, time.time() - started), flush=True)
        if lang not in LANGS:
            n_otherlang += 1
            continue
        key = link_key(link)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        if lang == "ja":
            text = " ".join(t for t in (clean(title), clean(description), clean(content)) if t)
        else:
            text = clean(content)
        if len(text) < 20:
            n_empty += 1
            continue
        split = "test" if pub_date >= test_start else ("val" if pub_date >= val_start else "train")
        batch.append((split, lang, text))
        if len(batch) >= BATCH_DOCS:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    for s in SPLITS:
        tok_files[s].close()
        lens[s].tofile(open(work / ("lens_%s.u32" % s), "wb"))
        langcodes[s].tofile(open(work / ("langs_%s.u8" % s), "wb"))
    stats["_meta"] = {
        "rows_scanned": n_rows, "dup_links_skipped": n_dup, "empty_skipped": n_empty,
        "other_lang_skipped": n_otherlang, "eos_id": eos, "limit": limit,
        "anomaly_retried": n_retried, "anomaly_skipped": n_anomaly_skipped,
        "snapshot_damage": damage,
        "elapsed_sec": round(time.time() - started),
    }
    (work / "encode_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats["_meta"], indent=2), flush=True)
    if damage["broken_lines"] or damage["gzip_crc_error"]:
        raise SystemExit(
            "中断: スナップショットが破損しています(壊れた行 %d / CRC 不一致 %s)。\n"
            "  この結果はシャード化せずに破棄してください。parse_dump.py からやり直しが必要です。"
            % (damage["broken_lines"], damage["gzip_crc_error"]))


def shard(work):
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    index = {"tokenizer": SP_MODEL.name, "dtype": "uint16", "shard_tokens": SHARD_TOKENS,
             "seed": SEED, "created": datetime.now().isoformat(timespec="seconds"),
             "boundaries": json.loads((work / "boundaries.json").read_text()),
             "shards": []}
    for split in SPLITS:
        lens = array("I")
        lens.frombytes((work / ("lens_%s.u32" % split)).read_bytes())
        langs = array("B")
        langs.frombytes((work / ("langs_%s.u8" % split)).read_bytes())
        offsets = array("Q", [0])
        for n in lens:
            offsets.append(offsets[-1] + n)
        order = list(range(len(lens)))
        if split == "train":
            rng.shuffle(order)  # 記事単位のシャッフル(非 IID 対策・配布要件)。val/test は走査順のまま
        src = open(work / ("tokens_%s.u16" % split), "rb")

        shard_no = 0
        buf, cur_tokens, cur_docs = [], 0, 0
        lang_tokens = {l: 0 for l in LANGS}

        def flush_shard():
            nonlocal shard_no, buf, cur_tokens, cur_docs, lang_tokens
            if cur_tokens == 0:
                return
            name = "%s_%03d.bin" % (split, shard_no)
            h = hashlib.sha256()
            with open(SHARDS_DIR / name, "wb") as f:
                for chunk in buf:
                    f.write(chunk)
                    h.update(chunk)
                f.flush()
                os.fsync(f.fileno())
                drop_cache(f)  # ページキャッシュではなく実際のディスクから読み直すため
            # 書いた直後に読み直して照合する(review16: 書き込み後のサイレント破損対策)
            back = hashlib.sha256()
            with open(SHARDS_DIR / name, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    back.update(chunk)
            if back.hexdigest() != h.hexdigest():
                raise SystemExit("中断: %s が書き込み後に化けています(sha256 不一致)。\n"
                                 "  RAM または書き込み経路の異常です。" % name)
            meta = {"name": name, "split": split, "tokens": cur_tokens, "docs": cur_docs,
                    "sha256": h.hexdigest(),
                    "langs": {l: n for l, n in lang_tokens.items() if n}}
            (SHARDS_DIR / (name + ".json")).write_text(json.dumps(meta, indent=2))
            index["shards"].append(meta)
            print("  wrote %s: %.1fM tokens, %d docs" % (name, cur_tokens / 1e6, cur_docs), flush=True)
            shard_no += 1
            buf, cur_tokens, cur_docs = [], 0, 0
            lang_tokens = {l: 0 for l in LANGS}

        for idx in order:
            n = lens[idx]
            if cur_tokens and cur_tokens + n > SHARD_TOKENS:  # 記事境界でのみ区切る(≤50M/シャード)
                flush_shard()
            src.seek(offsets[idx] * 2)
            buf.append(src.read(n * 2))
            cur_tokens += n
            cur_docs += 1
            lang_tokens[LANGS[langs[idx]]] += n
        flush_shard()
        src.close()
    (SHARDS_DIR / "shards_index.json").write_text(json.dumps(index, indent=2))
    total = sum(m["tokens"] for m in index["shards"])
    print("done: %d shards, %.2fB tokens total" % (len(index["shards"]), total / 1e9), flush=True)


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if cmd in ("encode", "all"):
        encode(work, limit)
    if cmd in ("shard", "all"):
        shard(work)
    if cmd not in ("encode", "shard", "all"):
        raise SystemExit("unknown command: %s" % cmd)


if __name__ == "__main__":
    main()
