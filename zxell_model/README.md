# zxell_model

English | [日本語](README.ja.md)

A GPT-style decoder-only Transformer implemented from scratch, with its training and evaluation code.
It is used in phase 1 (the single-machine baseline), and from phase 2 on the same code is reused for local training on the clients.

| File | Contents |
|---|---|
| `config.py` | The S / M / L / tiny configurations (layers, d_model, heads, ctx). 48k vocabulary; the embedding and the LM head share weights |
| `model.py` | RMSNorm (Pre-Norm), RoPE, causal self-attention, SwiGLU FFN, the GPT itself, and plain generation |
| `data.py` | Loads shards (uint16, EOS between articles). The training sampler and per-language perplexity evaluation |
| `train.py` | Training loop (AdamW, warmup + cosine, gradient clipping, bf16, checkpoints, periodic val evaluation) |
| `eval.py` | Per-language PPL of a checkpoint on val / test |
| `sample.py` | Generates text from a checkpoint (requires sentencepiece + `sp_bpe_48k.model`) |
| `fetch_shards.py` | Fetches shards from the coordinator API (`GET /api/shards/{name}`) and verifies their sha256 |

## Setup (RTX 3060 machine)

```bash
cd zxell_model
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install numpy sentencepiece
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Fetching shards (via the API)

```bash
export ZXELL_API_URL=http://192.168.1.2          # on the LAN, go through nginx (:80)
export ZXELL_API_HOST=api.zxell.ai               # Host header when addressing the server by IP
export ZXELL_API_KEY=<API key of an approved client>
python fetch_shards.py ~/zxell-shards                 # everything (about 12GB)
python fetch_shards.py ~/zxell-shards --only val,test # evaluation shards only
```

## Training

```bash
# S (≈30M) baseline. batch 32 × ctx 1024 = 32k tokens/step
python train.py --preset S --shards ~/zxell-shards --out runs/S_baseline --steps 20000
# Resume
python train.py --preset S --shards ~/zxell-shards --out runs/S_baseline --steps 20000 --resume runs/S_baseline/ckpt_last.pt
```

Logs go to `runs/<name>/log.jsonl` (loss / lr / grad_norm / tok_per_s, plus periodic per-language PPL on val).

## Evaluation and generation

```bash
python eval.py runs/S_baseline/ckpt_final.pt ~/zxell-shards test
python sample.py runs/S_baseline/ckpt_final.pt --sp ~/sp_bpe_48k.model --prompt "The central bank"
```

## CPU smoke test (server machine, no GPU)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
python train.py --preset tiny --shards ~/zxell-storage/shards --max-train-shards 1 \
    --out runs/smoke --steps 50 --batch 4 --eval-every 25 --eval-tokens 20000 --warmup 5
```
