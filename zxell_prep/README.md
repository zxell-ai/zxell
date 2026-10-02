# zxell_prep

フェーズ1 の前処理・トークナイザ関連スクリプト（サーバ機で実行する。クライアントには配布しない）。

- `extract_sample.py` — feed_items からトークナイザ学習・評価用サンプルを言語別に抽出
  （SELECT のみ。ja は増量サンプリング。1 記事 1 行・空白正規化・link 重複除去）
- `train_compare_tokenizers.py` — SentencePiece(BPE) 48k / 64k の学習と、
  言語別圧縮効率・コーパス総トークン数見積もりの比較レポート
- `parse_dump.py` — バックアップの mysqldump（.sql.gz）を直接ストリーミングパースして、
  前処理に必要な 7 列だけの JSONL スナップショットを作る。書き終えた gz は必ず読み直して
  gzip CRC・行数・全行の JSON 妥当性を検証し、合格したときだけ `.sha256` を出す
- `tokenize_full.py` — 全量トークン化+シャード化（48k 採用決定後の本番前処理）。
  スナップショット走査 → 正規化・重複除去 → sp_bpe_48k エンコード → pub_date で
  train/val/test を時系列分離 → 50M トークン/シャード（uint16 .bin + メタ JSON）を
  `~/zxell-storage/shards/` に出力

トークン化の読み取り元は稼働 MySQL ではなくバックアップのスナップショット。
2026-08-13 に稼働 DB の feed_items で InnoDB ページ破損が起きたため（review14）、
壊れかけのディスクを毎回 46GB 走査しない方針に変更した。

同日、書き出したスナップショット自体が書き込み後に化ける事象も起きている（review16）。
本機はサイレント破損の履歴があるため、**大きな出力は必ず書いた直後に読み直して検証する**
（`parse_dump.py` の `verify_output`、`tokenize_full.py` のシャード sha256 照合）。

実行環境（サーバ機）: `zxell_prep/.venv`（Python 3.14 + sentencepiece）。
2026-09 の新サーバ移行で稼働 MySQL は喪失したため、入力はダンプから作ったスナップショットのみ。
旧サーバ（RAM 故障機）が作った `/mnt/exssd/zxell/work/` とスナップショットは 2026-10-02 に削除済み。

| 用途 | 場所（既定値） | 環境変数 |
|---|---|---|
| コーパス原本 | `/mnt/exssd/zxell/backup/feed_items/sphered_tc20250330.sql.gz`（複製 `~/zxell-archive/`） | — |
| スナップショット | `~/zxell-archive/snapshot_20260905` | `ZXELL_SNAPSHOT` |
| トークナイザ作業 | `~/zxell-work/phase1`（`sp_bpe_48k.model` もここ） | `ZXELL_SP_MODEL` |
| 全量処理 | `~/zxell-work/phase1_full`（`boundaries.json` を置く） | — |
| シャード出力 | `~/zxell-storage/shards`（サーバの `ZXELL_STORAGE_DIR` 配下） | `ZXELL_SHARDS_DIR` |

```bash
PY=.venv/bin/python
WORK=~/zxell-work/phase1
SNAP=~/zxell-archive/snapshot_20260905

# ダンプ → スナップショット
$PY parse_dump.py /mnt/exssd/zxell/backup/feed_items/sphered_tc20250330.sql.gz $SNAP

# トークナイザ
$PY extract_sample.py $WORK $SNAP
$PY train_compare_tokenizers.py $WORK build
$PY train_compare_tokenizers.py $WORK train48
$PY train_compare_tokenizers.py $WORK train64
$PY train_compare_tokenizers.py $WORK report

# 全量前処理（境界 → エンコード → シャード）
$PY compute_boundaries.py ~/zxell-work/phase1_full $SNAP
$PY tokenize_full.py ~/zxell-work/phase1_full all
```
