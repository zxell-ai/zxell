# zxell_prep

English | [日本語](README.ja.md)

Phase-1 preprocessing and tokenizer scripts (run on the server machine; not distributed to clients).
In pipeline order:

- `parse_dump.py` — stream-parses the backup SQL dump (.sql.gz) directly and builds a JSONL snapshot
  containing only the 7 columns preprocessing needs. Once the gz is written it is always read back to
  verify the gzip CRC, the line count, and that every line is valid JSON; `.sha256` is written only if it passes
- `extract_sample.py` — scans the snapshot once and extracts per-language samples for tokenizer training and evaluation
  (5% for every language, raised to 25% for ja. One article per line, whitespace normalized, duplicate links removed)
- `train_compare_tokenizers.py` — trains SentencePiece (BPE) at 48k / 64k and produces a comparison report of
  per-language compression efficiency and the estimated total token count of the corpus (48k was adopted)
- `compute_boundaries.py` — computes the train / val / test split boundaries **per language** and writes `boundaries.json`
  (for each language, the newest 8,000 articles go to test and the 3,000 before them to val. Collection ended at
  different times for different languages, so a single cut-off date for all languages would leave val/test with one language only)
- `tokenize_full.py` — full tokenization + sharding (the production preprocessing).
  Scan the snapshot → normalize and deduplicate → encode with sp_bpe_48k → split into train/val/test according to
  `boundaries.json` → shuffle train at the article level → write 50M-token shards (uint16 .bin + metadata JSON)
- `textprep.py` — text cleanup shared by the two scripts above (whitespace normalization, HTML tag removal and
  entity decoding, and for ja the concatenation of title / description / body with duplicates dropped). It is shared so that
  the samples and the production run build text the same way
- `peek_shard.py` — a small tool that decodes a shard back to text, article by article (for eyeballing the training data)

The only input is the snapshot built from the backup dump (the database that used to be live no longer exists).

In the past, intermediate files were found corrupted after they had been written (a memory fault on the old server).
For that reason, **every large output must be read back and verified right after it is written**
(`verify_output` in `parse_dump.py`, the shard sha256 check in `tokenize_full.py`). Keep this pattern when adding new steps.

Runtime environment (server machine): `zxell_prep/.venv` (Python 3.14 + sentencepiece).

| Purpose | Location (default) | Environment variable |
|---|---|---|
| Original corpus | The backup SQL dump (kept in `~/zxell-archive/`, with a copy on a separate disk) | — |
| Snapshot | `~/zxell-archive/snapshot_20260905` | `ZXELL_SNAPSHOT` |
| Tokenizer work | `~/zxell-work/phase1` (`sp_bpe_48k.model` is also here) | `ZXELL_SP_MODEL` |
| Full processing | `~/zxell-work/phase1_full` (put `boundaries.json` here) | — |
| Shard output | `~/zxell-storage/shards` (under the server's `ZXELL_STORAGE_DIR`) | `ZXELL_SHARDS_DIR` |

```bash
PY=.venv/bin/python
WORK=~/zxell-work/phase1
SNAP=~/zxell-archive/snapshot_20260905

# Dump -> snapshot
$PY parse_dump.py ~/zxell-archive/<dump>.sql.gz $SNAP

# Tokenizer
$PY extract_sample.py $WORK $SNAP
$PY train_compare_tokenizers.py $WORK build
$PY train_compare_tokenizers.py $WORK train48
$PY train_compare_tokenizers.py $WORK train64
$PY train_compare_tokenizers.py $WORK report

# Full preprocessing (boundaries -> encode -> shards)
$PY compute_boundaries.py ~/zxell-work/phase1_full $SNAP
$PY tokenize_full.py ~/zxell-work/phase1_full all

# Eyeball the contents of a shard
$PY peek_shard.py train_000
```
