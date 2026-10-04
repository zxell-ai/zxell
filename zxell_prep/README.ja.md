# zxell_prep

[English](README.md) | 日本語

フェーズ1 の前処理・トークナイザ関連スクリプト（サーバ機で実行する。クライアントには配布しない）。
パイプラインの順に:

- `parse_dump.py` — バックアップの SQL ダンプ（.sql.gz）を直接ストリーミングパースして、
  前処理に必要な 7 列だけの JSONL スナップショットを作る。書き終えた gz は必ず読み直して
  gzip CRC・行数・全行の JSON 妥当性を検証し、合格したときだけ `.sha256` を出す
- `extract_sample.py` — スナップショットを 1 回走査して、トークナイザ学習・評価用のサンプルを言語別に抽出
  （全言語 5%、ja は 25% に増量。1 記事 1 行・空白正規化・link 重複除去）
- `train_compare_tokenizers.py` — SentencePiece(BPE) 48k / 64k の学習と、
  言語別圧縮効率・コーパス総トークン数見積もりの比較レポート（48k を採用）
- `compute_boundaries.py` — train / val / test の分割境界を**言語別に**算出して `boundaries.json` を出力
  （各言語の新しい側から test 8,000 記事、その直前 val 3,000 記事。言語によって収集の終了時期が違うため、
  全言語共通の日付で切ると val/test が 1 言語に偏る）
- `tokenize_full.py` — 全量トークン化+シャード化（本番前処理）。
  スナップショット走査 → 正規化・重複除去 → sp_bpe_48k エンコード → `boundaries.json` に従って
  train/val/test に分割 → train は記事単位でシャッフル → 50M トークン/シャード（uint16 .bin + メタ JSON）を出力
- `textprep.py` — 上の 2 スクリプトが共有するテキスト整形（空白正規化、HTML タグ除去と文字実体参照の復号、
  ja の title / description / 本文の連結と重複除去）。サンプルと本番で文の作り方を揃えるために共通化している
- `peek_shard.py` — シャードを記事単位で原文にデコードして表示する小道具（学習データの目視確認用）

入力は、バックアップのダンプから作ったスナップショットだけ（稼働していたデータベースは残っていない）。

過去に、書き出した中間ファイルが書き込み後に壊れていた事象があった（旧サーバのメモリ故障）。
そのため、**大きな出力は必ず書いた直後に読み直して検証する**
（`parse_dump.py` の `verify_output`、`tokenize_full.py` のシャード sha256 照合）。新しい工程を足すときもこの形を守る。

実行環境（サーバ機）: `zxell_prep/.venv`（Python 3.14 + sentencepiece）。

| 用途 | 場所（既定値） | 環境変数 |
|---|---|---|
| コーパス原本 | バックアップの SQL ダンプ（`~/zxell-archive/` に置き、別ディスクにも複製） | — |
| スナップショット | `~/zxell-archive/snapshot_20260905` | `ZXELL_SNAPSHOT` |
| トークナイザ作業 | `~/zxell-work/phase1`（`sp_bpe_48k.model` もここ） | `ZXELL_SP_MODEL` |
| 全量処理 | `~/zxell-work/phase1_full`（`boundaries.json` を置く） | — |
| シャード出力 | `~/zxell-storage/shards`（サーバの `ZXELL_STORAGE_DIR` 配下） | `ZXELL_SHARDS_DIR` |

```bash
PY=.venv/bin/python
WORK=~/zxell-work/phase1
SNAP=~/zxell-archive/snapshot_20260905

# ダンプ → スナップショット
$PY parse_dump.py ~/zxell-archive/<dump>.sql.gz $SNAP

# トークナイザ
$PY extract_sample.py $WORK $SNAP
$PY train_compare_tokenizers.py $WORK build
$PY train_compare_tokenizers.py $WORK train48
$PY train_compare_tokenizers.py $WORK train64
$PY train_compare_tokenizers.py $WORK report

# 全量前処理（境界 → エンコード → シャード）
$PY compute_boundaries.py ~/zxell-work/phase1_full $SNAP
$PY tokenize_full.py ~/zxell-work/phase1_full all

# シャードの中身を目視確認
$PY peek_shard.py train_000
```
