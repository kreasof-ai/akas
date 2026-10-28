"""LFM3 hybrid convolution, full attention and sliding attention geometry."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LFM3Config:
    width: int
    ff: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: tuple[str, ...]
    vocab: int
    conv: int
    epsilon: float
    max_context: int
    window: int
    theta: tuple[float, ...]
    rotary: tuple[int, ...]
    sandwich: bool = True
    tied: bool = True

    def __post_init__(self):
        if (
            any(
                type(x) is not int or x <= 0
                for x in (
                    self.width,
                    self.ff,
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                    self.vocab,
                    self.conv,
                    self.max_context,
                    self.window,
                )
            )
            or self.heads % self.kv_heads
            or self.head_dim not in (64, 128, 256)
            or self.conv < 2
            or self.ff % 32
            or not self.layers
            or set(self.layers) - {"conv", "full_attention", "sliding_attention"}
            or len(self.theta) != len(self.layers)
            or len(self.rotary) != len(self.layers)
            or any(
                type(x) is not int or x <= 0 or x > self.head_dim or x % 2
                for x in self.rotary
            )
            or any(not math.isfinite(x) or x <= 0 for x in (*self.theta, self.epsilon))
            or type(self.sandwich) is not bool
            or type(self.tied) is not bool
        ):
            raise ValueError("unsupported LFM3 geometry")

    @classmethod
    def from_huggingface(cls, data):
        if data.get("model_type") != "lfm3":
            raise ValueError("requires LFM3 config.json")
        if data.get("conv_bias", False) or data.get("rope_scaling"):
            raise ValueError("LFM3 requires bias-free convolution and default RoPE")
        layers = tuple(data["layer_types"])
        if len(layers) != data["num_hidden_layers"]:
            raise ValueError("LFM3 layer count differs from layer_types")
        d = data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])
        rope = data["rope_parameters"]
        theta, rotary = [], []
        for kind in layers:
            p = rope.get(kind, {}) if kind != "conv" else {}
            if p.get("rope_type", "default") != "default":
                raise ValueError("LFM3 requires default RoPE")
            rot = d * p.get("partial_rotary_factor", 1.0)
            if rot != int(rot):
                raise ValueError("LFM3 rotary dimensions must be integral")
            theta.append(p.get("rope_theta", 10000.0))
            rotary.append(int(rot))
        return cls(
            data["hidden_size"],
            data["intermediate_size"],
            data["num_attention_heads"],
            data["num_key_value_heads"],
            d,
            layers,
            data["vocab_size"],
            data["conv_L_cache"],
            data.get("norm_eps", 1e-5),
            data["max_position_embeddings"],
            data["sliding_window"],
            tuple(theta),
            tuple(rotary),
            data.get("use_sandwich_norm", True),
            data.get("tie_word_embeddings", True),
        )

    @classmethod
    def from_gguf(cls, gguf):
        m = gguf.metadata
        if m.get("general.architecture") != "lfm3":
            raise ValueError("requires lfm3 GGUF architecture")
        get = lambda k: m["lfm3." + k]
        count = get("block_count")
        kv, swa = (
            get("attention.head_count_kv"),
            get("attention.sliding_window_pattern"),
        )
        if (
            not isinstance(kv, list)
            or not isinstance(swa, list)
            or len(kv) != count
            or len(swa) != count
            or any(type(n) is not int or n < 0 for n in kv)
            or any(type(s) is not bool or (s and not n) for s, n in zip(swa, kv))
            or len(set(kv) - {0}) != 1
        ):
            raise ValueError("invalid LFM3 per-layer head/window metadata")
        d = get("attention.key_length")
        if get("attention.value_length") != d or get("rope.dimension_count") != d:
            raise ValueError(
                "LFM3 requires equal full-head key/value and rotary dimensions"
            )
        # GGUF stores full-head split-half pairs and disables the final frequencies
        # using large frequency factors. Keep this layout instead of rotating a leading subspace.
        import numpy as np

        factors = gguf.array("rope_freqs.weight")
        live = factors == 1
        n = int(live.sum())
        if (
            factors.shape != (d // 2,)
            or n < 1
            or not np.array_equal(live, np.arange(d // 2) < n)
            or not np.all(factors[:n] == 1)
            or not np.all(factors[n:] >= 1e30)
        ):
            raise ValueError("unsupported LFM3 rotary frequency factors")
        sandwich = "blk.0.post_attention_norm.weight" in gguf.tensors
        layers = tuple(
            "conv" if not k else "sliding_attention" if s else "full_attention"
            for k, s in zip(kv, swa)
        )
        return cls(
            get("embedding_length"),
            get("feed_forward_length"),
            get("attention.head_count"),
            next(k for k in kv if k),
            d,
            layers,
            gguf.tensors["token_embd.weight"].shape[0],
            get("shortconv.l_cache"),
            get("attention.layer_norm_rms_epsilon"),
            get("context_length"),
            get("attention.sliding_window"),
            tuple(
                get("rope.freq_base_swa") if s else get("rope.freq_base") for s in swa
            ),
            tuple(d if s else 2 * n for s in swa),
            sandwich,
            "output.weight" not in gguf.tensors,
        )
