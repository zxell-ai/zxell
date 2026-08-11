# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zxell is a distributed AI training project. The goal is to train a custom AI model on a large news-article corpus (~40–45GB: title, summary, category, body in English/German/French/Japanese) without high-end GPU hardware, by distributing computation across many low-spec clients. This repo contains the coordination server: clients GET pending tasks from it and POST computation results back.

Project planning documents (in Japanese) live in `_private/reviews/` (untracked).

## Running the Server

The server lives in `zxell_server/` and uses flat imports (`import models, schemas`), so it must be run from inside that directory:

```bash
cd zxell_server
pip install -r requirements.txt
export ZXELL_DB_URL="postgresql://user:password@localhost/zxell_db"  # required, no default
export ZXELL_ADMIN_API_KEY="..."                                     # required, no default
uvicorn main:app --reload
```

Settings are defined in `config.py` (pydantic-settings, `ZXELL_` prefix, optional `.env` file — see `.env.example`). Passing them as environment variables is the official procedure; never put real credentials in `config.py`. `ZXELL_DB_URL` must point at a PostgreSQL database; startup fails with a pydantic ValidationError if a required setting is unset. Tables are auto-created on startup via `models.Base.metadata.create_all` (no migrations tooling).

There are currently no tests or lint configuration.

## Architecture

Single FastAPI app (`zxell_server/main.py`) with a task-queue workflow over one table:

- `models.py` — SQLAlchemy model `TrainingData` (table `training_data`): article fields plus `status` (`pending` → `processing` → `done`) and `assigned_to` (client id).
- `schemas.py` — Pydantic I/O schemas: `TrainingDataOut` (task handed to a client), `ResultIn` (result posted back).
- `database.py` — engine/session setup from `ZXELL_DB_URL`.

API flow:
- `GET /api/get-task?client_id=...` — claims the first `pending` row: marks it `processing`, records `assigned_to`, returns it.
- `POST /api/post-result` — marks the given `data_id` as `done`. (The posted `result_json` is not yet persisted.)
