# zxell_server

English | [日本語](README.ja.md)

The coordination server for zxell distributed training (FastAPI + PostgreSQL).
Clients fetch tasks (with a lease) and submit their results.

## Setup

```bash
cd zxell_server
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The official way to pass settings is through environment variables (`ZXELL_` prefix).
`ZXELL_DB_URL` and `ZXELL_ADMIN_API_KEY` are required (the server fails at startup if either is unset).

```bash
export ZXELL_DB_URL="postgresql://zxell:PASSWORD@localhost/zxell_db"
export ZXELL_ADMIN_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
# Optional: export ZXELL_STORAGE_DIR=~/zxell-storage / export ZXELL_LEASE_SECONDS=3600
```

For development you may instead copy `.env.example` to `zxell_server/.env` and fill in real values (`.env` is gitignored).

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Tables are created automatically at startup (`Base.metadata.create_all`. There is no migration tool yet,
so adding a column to an existing table requires a manual ALTER).

## Authentication

- Client API: the `X-API-Key` header. A key is issued by `POST /api/clients/register`, but it is
  **inactive until an administrator approves it** (approval-gated).
- Admin API: the `X-Admin-Key` header (checked against `ZXELL_ADMIN_API_KEY`).

## API reference

| Method / path | Auth | Description |
|---|---|---|
| `POST /api/clients/register` | none | Registration request. Issues an API key (status=pending) |
| `GET /api/clients?status=pending` | admin | List clients (check who is awaiting approval) |
| `POST /api/clients/{id}/approve` | admin | Approve (activates the API key) |
| `POST /api/clients/{id}/disable` | admin | Disable (reject or force out) |
| `POST /api/tasks` | admin | Create tasks in bulk |
| `GET /api/tasks/next?types=train` | client | Lease the next task (filter by task type with `types`) |
| `POST /api/results` | client | Submit a result (multipart: task_id / base_weight_version / metrics / artifact) |
| `POST /api/weights` | admin | Register a global weight snapshot |
| `GET /api/weights/latest` | client | Metadata of the latest weights |
| `GET /api/weights/{version}` | client | Metadata of a given version |
| `GET /api/weights/{version}/download` | client | Download the weight file |
| `GET /api/shards/{name}` | client | Download a tokenized shard |
| `GET /api/status` | admin | Task counts, client list, and latest weights (for the dashboard) |

Task types are `preprocess` / `train` / `eval` / `verify`.
Submitting a `train` result requires `base_weight_version` (to handle staleness).

## curl examples

```bash
ADMIN='X-Admin-Key: <admin key>'

# Register a client (client side) -> save the returned api_key
curl -s -X POST localhost:8000/api/clients/register \
  -H 'Content-Type: application/json' \
  -d '{"name": "gpu-box-1", "capabilities": {"gpu": "RTX 3060", "vram_gb": 12}}'

# List pending clients -> approve (admin side)
curl -s -H "$ADMIN" 'localhost:8000/api/clients?status=pending'
curl -s -X POST -H "$ADMIN" localhost:8000/api/clients/<client_id>/approve

# Create tasks (admin side)
curl -s -X POST -H "$ADMIN" -H 'Content-Type: application/json' localhost:8000/api/tasks \
  -d '[{"type": "train", "payload": {"shard": "shard_00042.bin", "base_weight_version": 1, "local_steps": 200}}]'

# Fetch a task -> submit a result (client side)
KEY='X-API-Key: <api_key>'
curl -s -H "$KEY" 'localhost:8000/api/tasks/next?types=train'
curl -s -X POST -H "$KEY" localhost:8000/api/results \
  -F task_id=1 -F base_weight_version=1 \
  -F 'metrics={"loss": 2.31, "tokens": 409600}' \
  -F artifact=@delta.bin

# Status (admin side)
curl -s -H "$ADMIN" localhost:8000/api/status
```
