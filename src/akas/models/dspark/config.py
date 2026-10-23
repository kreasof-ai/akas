"""Geometry and target-feature contract for a GGUF DSpark draft sidecar."""

from dataclasses import dataclass
import math
from akas.checkpoints.gguf import GGUFError


@dataclass(frozen=True)
class DSparkDraftConfig:
    width: int
    ff: int
    heads: int
    kv_heads: int
    depth: int
    vocab: int
    epsilon: float
    theta: float
    max_context: int
    block_size: int
    markov_rank: int
    mask_token: int
    target_layers: tuple[int, ...]

    @property
    def head_dim(self):
        return self.width // self.heads

    @property
    def feature_width(self):
        return self.width * len(self.target_layers)

    @classmethod
    def from_gguf(cls, checkpoint, target):
        m, tensors = checkpoint.metadata, checkpoint.tensors
        if (
            m.get("general.architecture") != "dflash"
            or "markov_w1.weight" not in tensors
        ):
            raise GGUFError("DSpark requires a DFlash GGUF sidecar with a Markov head")
        if m.get("dflash.sample_from_anchor") is not True:
            raise GGUFError("DSpark requires the anchor-first draft block convention")
        layers = m.get("dflash.target_layers")
        if (
            not isinstance(layers, list)
            or not layers
            or any(
                type(i) is not int or not 1 <= i <= len(target.layers) for i in layers
            )
            or len(set(layers)) != len(layers)
        ):
            raise GGUFError(
                "DSpark target layers must be distinct target hidden-state indices"
            )
        # GGUF indexes the hidden-state tuple including the embedding at zero.
        # Public composition captures the output of zero-based target layers.
        rank_shape = tensors["markov_w1.weight"].shape
        if len(rank_shape) != 2:
            raise GGUFError("DSpark Markov embeddings must be a matrix")
        try:
            config = cls(
                m["dflash.embedding_length"],
                m["dflash.feed_forward_length"],
                m["dflash.attention.head_count"],
                m["dflash.attention.head_count_kv"],
                m["dflash.block_count"],
                target.vocab,
                m["dflash.attention.layer_norm_rms_epsilon"],
                m["dflash.rope.freq_base"],
                min(m["dflash.context_length"], target.max_context),
                m["dflash.block_size"],
                rank_shape[1],
                m["tokenizer.ggml.mask_token_id"],
                tuple(i - 1 for i in layers),
            )
        except KeyError as error:
            raise GGUFError(
                "DSpark sidecar is missing required configuration"
            ) from error
        if (
            any(
                type(v) is not int or v <= 0
                for v in (
                    config.width,
                    config.ff,
                    config.heads,
                    config.kv_heads,
                    config.depth,
                    config.vocab,
                    config.max_context,
                    config.block_size,
                    config.markov_rank,
                )
            )
            or config.width != target.width
            or config.width % config.heads
            or config.heads % config.kv_heads
            or config.head_dim != 64
            or config.ff % 32
            or type(config.mask_token) is not int
            or not 0 <= config.mask_token < config.vocab
            or not all(
                math.isfinite(v) and v > 0 for v in (config.epsilon, config.theta)
            )
        ):
            raise GGUFError("Unsupported DSpark draft or incompatible target geometry")
        if len(m.get("tokenizer.ggml.tokens", ())) != config.vocab:
            raise GGUFError("DSpark and target vocabularies differ")
        return config
