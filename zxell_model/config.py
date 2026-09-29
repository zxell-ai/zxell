"""モデル構成(roadmap 3.2 の S / M / L と、CPU スモーク用の tiny)。"""

from dataclasses import dataclass, asdict

VOCAB_SIZE = 48000   # sp_bpe_48k(review13 / tokenizer_report.md)
EOS_ID = 2
LANGS = ("en", "de", "fr", "ja")   # シャードの langs 副ファイルの番号 → 言語(tokenize_full.py と同順)


@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    ctx: int
    vocab_size: int = VOCAB_SIZE
    rope_theta: float = 10000.0
    tie_embeddings: bool = True   # 埋め込みと LM Head の重み共有(roadmap 3.2)

    @property
    def head_dim(self):
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    @property
    def ffn_hidden(self):
        # SwiGLU は 3 本の射影を持つので、パラメータ量を 4*d の FFN と揃える 8/3*d を 64 の倍数に丸める
        h = int(self.d_model * 8 / 3)
        return ((h + 63) // 64) * 64

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


PRESETS = {
    "tiny": ModelConfig(n_layer=2, d_model=64, n_head=2, ctx=128),        # CPU スモーク用
    "S": ModelConfig(n_layer=6, d_model=384, n_head=6, ctx=1024),         # ≈30M
    "M": ModelConfig(n_layer=12, d_model=768, n_head=12, ctx=1024),       # ≈125M
    "L": ModelConfig(n_layer=24, d_model=1280, n_head=20, ctx=2048),      # ≈500M
}
