# zxell_model

イチから実装する GPT 系デコーダオンリー Transformer と、その学習・評価コード。
フェーズ 1(単一マシンのベースライン)で使い、フェーズ 2 以降はクライアントの局所学習にも同じコードを流用する。

| ファイル | 内容 |
|---|---|
| `config.py` | S / M / L / tiny の構成(層数・d_model・head・ctx)。語彙 48k、埋め込みと LM Head は重み共有 |
| `model.py` | RMSNorm(Pre-Norm)、RoPE、causal Self-Attention、SwiGLU FFN、GPT 本体、素朴な生成 |
| `data.py` | シャード(uint16・記事間 EOS)の読み込み。学習サンプラと、言語別パープレキシティ評価 |
| `train.py` | 学習ループ(AdamW・warmup+cosine・勾配クリップ・bf16・チェックポイント・定期 val 評価) |
| `eval.py` | チェックポイントの val / test 言語別 PPL |
| `sample.py` | チェックポイントから文章生成(要 sentencepiece + `sp_bpe_48k.model`) |
| `fetch_shards.py` | コーディネータ API(`GET /api/shards/{name}`)からシャードを取得・sha256 照合 |

## セットアップ(RTX 3060 機)

```bash
cd zxell_model
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install numpy sentencepiece
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## シャード取得(API 経由)

```bash
export ZXELL_API_URL=http://192.168.1.2          # LAN 内は nginx(:80)経由
export ZXELL_API_HOST=api.zxell.ai               # IP 直打ち時の Host ヘッダ
export ZXELL_API_KEY=<承認済みクライアントの API キー>
python fetch_shards.py ~/zxell-shards                 # 全部(約 12GB)
python fetch_shards.py ~/zxell-shards --only val,test # 評価用だけ
```

## 学習

```bash
# S(≈30M)ベースライン。batch 32 × ctx 1024 = 32k トークン/ステップ
python train.py --preset S --shards ~/zxell-shards --out runs/S_baseline --steps 20000
# 再開
python train.py --preset S --shards ~/zxell-shards --out runs/S_baseline --steps 20000 --resume runs/S_baseline/ckpt_last.pt
```

ログは `runs/<name>/log.jsonl`(loss / lr / grad_norm / tok_per_s と、定期的な val の言語別 PPL)。

## 評価・生成

```bash
python eval.py runs/S_baseline/ckpt_final.pt ~/zxell-shards test
python sample.py runs/S_baseline/ckpt_final.pt --sp ~/sp_bpe_48k.model --prompt "The central bank"
```

## CPU スモーク(サーバ機・GPU なし)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
python train.py --preset tiny --shards ~/zxell-storage/shards --max-train-shards 1 \
    --out runs/smoke --steps 50 --batch 4 --eval-every 25 --eval-tokens 20000 --warmup 5
```
