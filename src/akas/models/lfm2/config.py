"""LFM2 architecture dimensions and supported GGUF geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    width: int
    ff: int
    heads: int
    kv_heads: int
    layers: tuple[str, ...]
    vocab: int
    conv: int
    epsilon: float
    theta: float
    max_context: int

    @property
    def head_dim(self):
        return self.width // self.heads

    @classmethod
    def from_gguf(cls, gguf):
        return parse_config(gguf)


import math
from akas.checkpoints.gguf import GGUFError


def parse_config(gguf):
    m = gguf.metadata
    if m.get("general.architecture") != "lfm2":
        raise GGUFError("requires the LFM2 architecture")
    get = lambda name: m["lfm2." + name]
    kv = get("attention.head_count_kv")
    if not isinstance(kv, list) or len(kv) != get("block_count"):
        raise GGUFError("requires per-layer attention metadata")
    unique = set(kv) - {0}
    if len(unique) != 1:
        raise GGUFError("requires uniform nonzero GQA head count")
    config = Config(
        get("embedding_length"),
        get("feed_forward_length"),
        get("attention.head_count"),
        unique.pop(),
        tuple(("attention" if n else "conv" for n in kv)),
        get("vocab_size"),
        get("shortconv.l_cache"),
        get("attention.layer_norm_rms_epsilon"),
        get("rope.freq_base"),
        get("context_length"),
    )
    if any(
        (
            type(value) is not int or value <= 0
            for value in (
                config.width,
                config.ff,
                config.heads,
                config.kv_heads,
                config.vocab,
                config.max_context,
            )
        )
    ):
        raise GGUFError("LFM2 dimensions must be positive integers")
    if not config.layers or any((type(n) is not int or n < 0 for n in kv)):
        raise GGUFError("invalid LFM2 layer/head metadata")
    if not all(
        (math.isfinite(value) and value > 0 for value in (config.epsilon, config.theta))
    ):
        raise GGUFError("LFM2 normalization/RoPE constants must be finite and positive")
    if (
        config.width % config.heads
        or config.heads % config.kv_heads
        or config.head_dim != 64
        or config.ff % 32
        or config.vocab % 4
        or (config.conv != 3)
    ):
        raise GGUFError("unsupported LFM2 head/convolution configuration")
    from akas.models.lfm2.weights import validate_tensors

    validate_tensors(config, gguf)
    return config
