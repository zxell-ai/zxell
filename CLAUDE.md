# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zxell is a distributed AI training project: a multilingual news-article corpus (~8M articles / ~6.5B tokens in English/German/French/Japanese) is used to train a from-scratch GPT-style language model, with computation distributed across low-spec client machines instead of high-end GPUs. This monorepo contains:

- `zxell_server/` — the coordination server (FastAPI). Clients register, lease tasks, and post results; the server also versions/distributes model weights and training shards.
- `zxell_client/` — client v1 (Windows/Linux). Register → wait for approval → lease task → process → post result. Task processing is still a dummy loop; real training lands in phase 2.
- `zxell_prep/` — phase-1 data preparation scripts (see below).
- `website/` — static landing pages for https://zxell.ai (`index.html` English, `ja/index.html` Japanese), deployed by copying to `/var/www/zxell.ai/`.

Project planning documents (in Japanese) live in `_private/` (untracked): `_private/reviews/` is the running review correspondence, `_private/reports/roadmap.md` is the canonical roadmap and decision log.

## Running the Server

The server lives in `zxell_server/` and uses flat imports (`import models, schemas`), so it must be run from inside that directory:

```bash
cd zxell_server
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ZXELL_DB_URL="postgresql://user:password@localhost/zxell_db"  # required, no default
export ZXELL_ADMIN_API_KEY="..."                                     # required, no default
.venv/bin/uvicorn main:app --reload
```

Settings are defined in `config.py` (pydantic-settings, `ZXELL_` prefix, optional `.env` file — see `.env.example`). Other settings: `ZXELL_STORAGE_DIR` (shards/weights/artifacts directory, default `storage`), `ZXELL_LEASE_SECONDS` (task lease, default 3600). Passing them as environment variables is the official procedure; never put real credentials in `config.py`. `ZXELL_DB_URL` must point at PostgreSQL; startup fails with a pydantic ValidationError if a required setting is unset. Tables are auto-created on startup via `models.Base.metadata.create_all` (no migrations tooling).

Production runs on the home server as systemd unit `zxell-server` (uvicorn on 127.0.0.1:8000, `EnvironmentFile=/etc/zxell/env`), behind nginx TLS at `https://api.zxell.ai`. `requirements.txt` is pinned for the production Python (currently 3.14).

There are currently no tests or lint configuration.

## Architecture

Single FastAPI app (`zxell_server/main.py`) over four tables (`models.py`):

- `clients` — approval-gated workers (`pending` → `approved` / `disabled`), per-client `api_key`, `capabilities` JSON, `trust_score`.
- `tasks` — leased work queue: `type` (`preprocess` / `train` / `eval` / `verify`), JSON `payload`, `status` (`pending` → `leased` → `done`/`failed`), lease expiry with automatic re-queue and `max_attempts`.
- `results` — posted results: `metrics` JSON, optional uploaded artifact (stored under `ZXELL_STORAGE_DIR`, path in `artifact_path`), and for `train` results a required `base_weight_version` (staleness handling).
- `weight_versions` — versioned global weights with checksum, uploaded/downloaded via the API.

Auth (`auth.py`): every client API requires the `X-API-Key` header of an approved client (401/403 otherwise); admin APIs (client approval, task creation, weight upload, `/api/status`) require `X-Admin-Key`. `/dashboard` serves a static admin dashboard (`dashboard.html`) that calls `/api/status` with the admin key.

Key endpoints: `POST /api/clients/register`, `POST /api/admin/clients/{id}/approve`, `POST /api/admin/tasks`, `GET /api/tasks/next` (leases one task, `FOR UPDATE SKIP LOCKED`), `POST /api/results`, `POST /api/admin/weights` / `GET /api/weights/latest|{version}|{version}/download`, `GET /api/shards/{name}`, `GET /api/status`.

## Data preparation (`zxell_prep/`)

The corpus source of truth is the 2025-03-30 MySQL dump `sphered_tc20250330.sql.gz` (the live MySQL was lost with the old server's disk in 2026-09). Pipeline order:

1. `parse_dump.py <dump.sql.gz> <outdir>` — stream-parse the dump into a verified JSONL snapshot (re-reads what it wrote; fails on any corruption).
2. `extract_sample.py <outdir> [snapshot_dir]` — sample corpus for tokenizer work (reads the snapshot).
3. `train_compare_tokenizers.py <workdir> build|train48|train64|report` — SentencePiece BPE training and 48k-vs-64k comparison (48k adopted).
4. `tokenize_full.py <workdir> encode|shard|all [limit]` — full tokenization and sharding; paths overridable via `ZXELL_SNAPSHOT`, `ZXELL_SP_MODEL`, `ZXELL_SHARDS_DIR`; requires `boundaries.json` (time-based train/val/test split) in the workdir.

All long steps verify their own output (write-then-re-read hashing) — keep that pattern for new steps; it has caught real silent corruption before.

## Conventions

- Commits are authored as `zxell-ai <214323922+zxell-ai@users.noreply.github.com>` (repo-local git config).
- Never commit real credentials, `_private/`, model weights, shards, or DB dumps. Only code and the website are public (GPL-3.0).
