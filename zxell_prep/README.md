# zxell_prep

フェーズ1 の前処理・トークナイザ関連スクリプト（サーバ機で実行する。クライアントには配布しない）。

- `extract_sample.py` — feed_items からトークナイザ学習・評価用サンプルを言語別に抽出
  （SELECT のみ。ja は増量サンプリング。1 記事 1 行・空白正規化・link 重複除去）
- `train_compare_tokenizers.py` — SentencePiece(BPE) 48k / 64k の学習と、
  言語別圧縮効率・コーパス総トークン数見積もりの比較レポート

実行環境（サーバ機）: `/mnt/exssd/zxell/work/venv`（pymysql + sentencepiece）。
作業ディレクトリ: `/mnt/exssd/zxell/work/phase1`。
MySQL 接続は `~/.my.cnf` を参照し、認証情報をコードに書かない。

```bash
VENV=/mnt/exssd/zxell/work/venv/bin/python
WORK=/mnt/exssd/zxell/work/phase1
$VENV extract_sample.py $WORK
$VENV train_compare_tokenizers.py $WORK build
$VENV train_compare_tokenizers.py $WORK train48
$VENV train_compare_tokenizers.py $WORK train64
$VENV train_compare_tokenizers.py $WORK report
```
