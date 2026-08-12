"""フェーズ1: SentencePiece(BPE)トークナイザの学習と語彙サイズ比較。

extract_sample.py の出力から言語バランスを調整した学習ファイルを作り、
語彙 48k / 64k の 2 モデルを学習 → 言語別の評価セット（学習に未使用）で
圧縮効率（文字数/トークン）を計測し、フェーズ0 の文字数実測からコーパス全体の
トークン数を見積もる。

使い方:
  python train_compare_tokenizers.py <workdir> build   # 学習ファイル生成
  python train_compare_tokenizers.py <workdir> train48 # 48k モデル学習
  python train_compare_tokenizers.py <workdir> train64 # 64k モデル学習
  python train_compare_tokenizers.py <workdir> report  # 比較レポート（JSON を標準出力）
"""

import json
import random
import sys
from pathlib import Path

import sentencepiece as spm

# 学習ファイルの言語配分（文字数上限）。自然比率のままだと ja が 1.5% しか入らず
# 日本語の語彙が育たないため、tokenizer 学習に限り ja を厚くする（学習データ配分とは別の話）
TRAIN_CHAR_BUDGET = {
    "en": 120_000_000,
    "de": 85_000_000,
    "fr": 70_000_000,
    "ja": 65_000_000,  # サンプルに 65M 字なければ有るだけ使う
}
MAX_LINE_BYTES = 12_000  # SentencePiece の max_sentence_length に収める（超過分は切り詰め）

# フェーズ0 実測（_private/reports/phase0_inventory.md）: 言語別 content_text 総文字数と
# link 重複除去後の残存率。ja は title+description 連結の増分を eval サンプルから補正する
PHASE0_CHARS = {"en": 17.08e9, "de": 6.02e9, "fr": 2.70e9, "ja": 0.40e9}
PHASE0_DOCS = {"en": 3_984_113, "de": 2_036_736, "fr": 1_116_522, "ja": 848_917}
DEDUP_KEEP = 0.947


def build(work):
    rng = random.Random(20260812)
    lines = []
    used = {}
    for lang, budget in TRAIN_CHAR_BUDGET.items():
        chars = 0
        src = (work / ("train_%s.txt" % lang)).open(encoding="utf-8")
        for line in src:
            line = line.rstrip("\n")
            b = line.encode("utf-8")
            if len(b) > MAX_LINE_BYTES:
                line = b[:MAX_LINE_BYTES].decode("utf-8", errors="ignore")
            lines.append(line)
            chars += len(line)
            if chars >= budget:
                break
        src.close()
        used[lang] = chars
    rng.shuffle(lines)
    out = work / "sp_train.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"lines": len(lines), "chars_used": used}, indent=2))


def train(work, vocab_size):
    spm.SentencePieceTrainer.train(
        input=str(work / "sp_train.txt"),
        model_prefix=str(work / ("sp_bpe_%dk" % (vocab_size // 1000))),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,          # 未知文字はバイト列へ（UNK を出さない。LLaMA 系と同じ）
        split_digits=True,           # 数字は 1 桁ずつ（ニュースは数値が多く、汎化に有利）
        max_sentence_length=16384,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        num_threads=2,
        normalization_rule_name="nmt_nfkc",
    )


def report(work):
    out = {"models": {}}
    for tag in ("48k", "64k"):
        model_path = work / ("sp_bpe_%s.model" % tag)
        if not model_path.exists():
            continue
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        langs = {}
        est_total = 0.0
        for lang in ("en", "de", "fr", "ja"):
            chars = tokens = docs = 0
            for line in (work / ("eval_%s.txt" % lang)).open(encoding="utf-8"):
                line = line.rstrip("\n")
                chars += len(line)
                tokens += len(sp.encode(line))
                docs += 1
            cpt = chars / tokens
            # コーパス全体のトークン数見積もり。ja は連結後の平均文字数 × 記事数で総文字数を出す
            if lang == "ja":
                total_chars = (chars / docs) * PHASE0_DOCS[lang]
            else:
                total_chars = PHASE0_CHARS[lang]
            est = total_chars * DEDUP_KEEP / cpt
            est_total += est
            langs[lang] = {
                "eval_docs": docs,
                "chars_per_token": round(cpt, 3),
                "est_corpus_tokens_B": round(est / 1e9, 2),
            }
        out["models"][tag] = {"langs": langs, "est_total_tokens_B": round(est_total / 1e9, 2)}
    print(json.dumps(out, indent=2))


def main():
    work = Path(sys.argv[1])
    cmd = sys.argv[2]
    if cmd == "build":
        build(work)
    elif cmd == "train48":
        train(work, 48000)
    elif cmd == "train64":
        train(work, 64000)
    elif cmd == "report":
        report(work)
    else:
        raise SystemExit("unknown command: %s" % cmd)


if __name__ == "__main__":
    main()
