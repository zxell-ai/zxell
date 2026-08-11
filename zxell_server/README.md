# zxell_server

zxell 分散学習の調整サーバ（FastAPI + PostgreSQL）。
クライアントはタスクを取得し（リース付き）、計算結果を提出する。

## セットアップ

```bash
cd zxell_server
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

設定は環境変数（`ZXELL_` プレフィックス）で渡すのが正式手順。
`ZXELL_DB_URL` と `ZXELL_ADMIN_API_KEY` は必須（未設定だと起動時にエラー）。

```bash
export ZXELL_DB_URL="postgresql://zxell:PASSWORD@localhost/zxell_db"
export ZXELL_ADMIN_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
# 任意: export ZXELL_STORAGE_DIR=/mnt/exssd/zxell/storage / export ZXELL_LEASE_SECONDS=3600
```

開発時は `.env.example` を `zxell_server/.env` にコピーして実値を書いてもよい（`.env` は gitignore 済み）。

起動:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

テーブルは起動時に自動作成される（`Base.metadata.create_all`。マイグレーションツールは未導入のため、
既存テーブルへの列追加は手動 ALTER が必要）。

## 認証

- クライアント API: `X-API-Key` ヘッダ。`POST /api/clients/register` で発行されるが、
  **管理者が承認するまで無効**（承認制）。
- 管理 API: `X-Admin-Key` ヘッダ（`ZXELL_ADMIN_API_KEY` と照合）。

## API 一覧

| メソッド/パス | 認証 | 説明 |
|---|---|---|
| `POST /api/clients/register` | なし | 登録申請。API キー発行（status=pending） |
| `GET /api/clients?status=pending` | 管理 | クライアント一覧（承認待ちの確認） |
| `POST /api/clients/{id}/approve` | 管理 | 承認（API キー有効化） |
| `POST /api/clients/{id}/disable` | 管理 | 無効化（却下・強制離脱） |
| `POST /api/tasks` | 管理 | タスク一括投入 |
| `GET /api/tasks/next?types=train` | クライアント | リース付きタスク払い出し（types で種別絞り込み） |
| `POST /api/results` | クライアント | 結果提出（multipart: task_id / base_weight_version / metrics / artifact） |
| `POST /api/weights` | 管理 | グローバル重みスナップショット登録 |
| `GET /api/weights/latest` | クライアント | 最新重みのメタデータ |
| `GET /api/weights/{version}` | クライアント | 指定バージョンのメタデータ |
| `GET /api/weights/{version}/download` | クライアント | 重みファイル取得 |
| `GET /api/shards/{name}` | クライアント | トークン化済みシャード取得 |
| `GET /api/status` | 管理 | タスク集計・クライアント一覧・最新重み（ダッシュボード用） |

タスク種別は `preprocess` / `train` / `eval` / `verify`。
`train` の結果提出には `base_weight_version` が必須（ステイルネス対策）。

## curl 例

```bash
ADMIN='X-Admin-Key: <管理キー>'

# クライアント登録（クライアント側）→ 返ってきた api_key を保存
curl -s -X POST localhost:8000/api/clients/register \
  -H 'Content-Type: application/json' \
  -d '{"name": "gpu-box-1", "capabilities": {"gpu": "RTX 3060", "vram_gb": 12}}'

# 承認待ち一覧 → 承認（管理側）
curl -s -H "$ADMIN" 'localhost:8000/api/clients?status=pending'
curl -s -X POST -H "$ADMIN" localhost:8000/api/clients/<client_id>/approve

# タスク投入（管理側）
curl -s -X POST -H "$ADMIN" -H 'Content-Type: application/json' localhost:8000/api/tasks \
  -d '[{"type": "train", "payload": {"shard": "shard_00042.bin", "base_weight_version": 1, "local_steps": 200}}]'

# タスク取得 → 結果提出（クライアント側）
KEY='X-API-Key: <api_key>'
curl -s -H "$KEY" 'localhost:8000/api/tasks/next?types=train'
curl -s -X POST -H "$KEY" localhost:8000/api/results \
  -F task_id=1 -F base_weight_version=1 \
  -F 'metrics={"loss": 2.31, "tokens": 409600}' \
  -F artifact=@delta.bin

# 稼働状況（管理側）
curl -s -H "$ADMIN" localhost:8000/api/status
```
