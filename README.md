# zxell.ai

**Training a language model from scratch on a multilingual news corpus — without high-end GPUs — by distributing the work across many low-spec volunteer machines.**

zxell.ai is an experiment in low-budget, low-communication distributed training. The corpus is a private archive of ~8 million news articles (title, summary, category, body) in English, German, French, and Japanese — roughly 6.5 billion tokens after deduplication. Instead of renting a GPU cluster, training is coordinated by a small central server that hands out tasks to ordinary consumer machines over HTTPS.

The deliverable is not just the model: it is **quantitative insight into how far a from-scratch model can get on commodity hardware**, measured against a single-machine baseline (perplexity, summarization quality, factual QA, scaling curves, communication cost).

## How it works

Hub-and-spoke, in the spirit of volunteer computing (BOINC / Folding@home):

```
[Central server]
  ├─ Coordinator API (FastAPI): client registration & approval, task leasing, result intake
  ├─ Parameter store: versioned global weight snapshots (checksummed)
  ├─ Shard storage: tokenized, packed training shards
  └─ Metadata DB (PostgreSQL): clients / tasks / results / weight versions

[Clients] (heterogeneous, low-spec)
  ├─ GPU workers: GET latest weights + a shard → train H local steps → POST a compressed delta
  └─ CPU workers: preprocessing / evaluation / verification tasks
```

Synchronous data parallelism is impossible over home connections (a 500M-parameter gradient is ~1 GB per step), so training follows the **DiLoCo / Local SGD** family: each client trains locally for hundreds of steps and submits only a compressed weight delta; the server aggregates deltas with an outer optimizer (Nesterov momentum) and publishes a new weight version. This cuts communication by 100–500× and tolerates clients joining, leaving, or failing mid-task (tasks are leased and automatically re-queued).

## Model

A decoder-only GPT-style Transformer, implemented from scratch (PyTorch as the base framework, no pretrained weights), with a modern recipe: RoPE positions, RMSNorm (pre-norm), SwiGLU FFN, tied embedding / LM head. The tokenizer is a custom multilingual BPE (48k–64k vocabulary) trained on the mixed corpus.

Three sizes, trained in order:

| Size | Params | Layout | Context | Role |
|---|---|---|---|---|
| S | ~30M | 6 layers, d=384 | 1024 | pipeline validation, smoke tests |
| M | ~125M | 12 layers, d=768 | 1024 | main testbed for comparing distribution schemes |
| L | ~500M | 24 layers, d=1280 | 2048 | final target, full corpus |

Pretraining is next-token prediction, followed by summarization SFT using the articles' own summaries as supervision. Images that accompany the articles are out of scope for phase one.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Data inventory (counts, languages, duplication, quality), legal policy, server hardening & public HTTPS endpoint | ✅ done |
| 1 | Tokenizer training, preprocessing pipeline, single-machine S-model baseline, evaluation harness | ⏳ next |
| 2 | Distributed MVP: client v1, FedAvg training of the S model across machines, ops dashboard | planned |
| 3 | Low-communication training (DiLoCo-style outer optimizer, 8-bit / top-k delta compression), verification & trust scoring, scheme comparison experiments | planned |
| 4 | Full L-model training, summarization SFT, QA evaluation, checkpoint archaeology, model-internals visualization | planned |
| 5 | Public technical report; decide on extensions (multimodal, open participation, WebGPU clients) | planned |

## Repository layout

- `zxell_server/` — the coordination server (FastAPI + SQLAlchemy + PostgreSQL). Implemented and running. See [`zxell_server/README.md`](zxell_server/README.md) for setup, configuration, and the full API reference (in Japanese).
- `zxell_client/` — the training/preprocessing client (planned).

## Running the server

```bash
cd zxell_server
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

export ZXELL_DB_URL="postgresql://user:password@localhost/zxell_db"  # required
export ZXELL_ADMIN_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"  # required
# optional: ZXELL_STORAGE_DIR, ZXELL_LEASE_SECONDS

uvicorn main:app --host 0.0.0.0 --port 8000
```

Settings use pydantic-settings with a `ZXELL_` prefix (a `.env` file also works; see `.env.example`). Never put real credentials in `config.py`. Tables are auto-created on startup. The server uses flat imports, so run it from inside `zxell_server/`.

API workflow in short: clients register (`POST /api/clients/register`) and wait for admin approval; approved clients lease tasks (`GET /api/tasks/next`), fetch shards and weights, and submit results with metrics and artifacts (`POST /api/results`). Admin endpoints (task submission, weight registration, approval, `/api/status`) require a separate admin key. Task types are `preprocess` / `train` / `eval` / `verify`.

## Participation & data

The project currently runs **closed**: clients are machines operated by the project itself, and the news corpus is not redistributed — only tokenized, shuffled, packed shards ever leave the server, and only to approved clients. Model weights and data statistics are not published at this stage; the code in this repository is the public part.

## License

[GPL-3.0](LICENSE)
