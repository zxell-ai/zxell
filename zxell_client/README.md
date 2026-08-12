# zxell_client

zxell 分散学習のクライアント v1（Windows / Linux 対応）。
サーバに登録 → 管理者の承認待ち → タスク取得（リース付き）→ 処理 → 結果提出、を繰り返す。

v1 の処理は**ダミー**（`train` は重みのダウンロードと checksum 検証まで行い、ダミー Δ を提出。
その他は 1 秒待って完了報告）。分散基盤の疎通・承認フロー・リース・リトライの検証が目的で、
実際の学習はフェーズ2 以降で実装する。

## セットアップ

### Windows

```bat
cd zxell_client
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python client.py
```

### Linux

```bash
cd zxell_client
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python client.py
```

初回実行で登録申請され、API キーが `~/.zxell/client.json`
（Windows は `C:\Users\<name>\.zxell\client.json`）に保存される。
管理者が承認するまで「承認待ち」を表示してポーリングを続け、承認されると自動で稼働を開始する。
再登録したいときはこのファイルを削除する。

## オプション・環境変数

| 引数 | 環境変数 | 既定値 | 説明 |
|---|---|---|---|
| `--server` | `ZXELL_SERVER_URL` | `https://api.zxell.ai` | サーバ URL |
| `--name` | `ZXELL_CLIENT_NAME` | ホスト名 | 登録名 |
| `--types` | `ZXELL_CLIENT_TYPES` | （空 = 全種別） | 希望タスク種別。GPU 機は `train`、CPU 機は `preprocess,eval,verify` など |
| `--state-file` | `ZXELL_STATE_FILE` | `~/.zxell/client.json` | API キーの保存先 |
| `--poll` | `ZXELL_POLL_SECONDS` | 30 | タスク無し/承認待ち時のポーリング間隔（秒） |
| `--once` | — | — | キューが空になったら終了（動作確認用） |

通信エラー・サーバ 5xx は指数バックオフ（最大 60 秒 × 5 回）で自動リトライする。
回線 IP の変動による接続断（最大 70 分程度）もこのリトライで吸収する。

## サーバと同一 LAN 内から使う場合の注意

ルーターがヘアピン NAT に対応していないため、LAN 内のマシンから
`https://api.zxell.ai` に直接つながらないことがある。その場合は hosts ファイルに
サーバの LAN アドレスを 1 行追加する（TLS 証明書はそのまま有効）:

- Windows: `C:\Windows\System32\drivers\etc\hosts`（管理者権限のメモ帳で編集）
- Linux: `/etc/hosts`

```
192.168.10.5 api.zxell.ai
```

サーバマシン自身で動かす場合は `127.0.0.1 api.zxell.ai` または
`--server http://127.0.0.1:8000` を使う。
