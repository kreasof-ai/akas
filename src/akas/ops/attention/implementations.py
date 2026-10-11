"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants
from akas.ops.kv_quantization import is_quantized


def quantized_cache(p):
    return is_quantized(p.get("kv_dtype", "bfloat16"))


IMPLEMENTATIONS = (
    *variants(
        ("attention", "attention_kv"),
        "attention.cuda.quantized",
        "block-fp8",
        phase="prefill",
        priority=30,
        accepts=quantized_cache,
    ),
    *variants(
        ("attention_partial", "attention_kv"),
        "attention.cuda.quantized",
        "block-fp8",
        phase="decode",
        priority=30,
        accepts=quantized_cache,
    ),
    *variants(
        ("attention_partial", "attention_grouped", "attention_merge"),
        "attention.cuda.packed",
        "packed",
        priority=20,
    ),
    *variants(
        ("qkv", "attention"), "attention.cuda.packed_baseline", "packed-baseline"
    ),
    *variants(
        ("qkv", "attention"), "attention.cuda.packed_baseline", "packed", priority=0
    ),
    *variants(
        ("qkv", "attention"),
        "attention.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
    *variants(
        ("attention_scores", "attention", "qkv"),
        "attention.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
    ),
    *variants(("attention_qkv", "attention"), "attention.cuda.bf16_batch", "bf16"),
    *variants(
        ("attention_q", "attention_kv", "attention_partial", "attention_merge"),
        "attention.cuda.decode",
        "block-fp8",
        phase="decode",
    ),
    *variants(
        ("attention_kv", "attention"),
        "attention.cuda.prefill",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
    ),
)

IMPLEMENTATIONS += tuple(
    item for provider in ("cuda", "webgpu")
    for item in variants(("rope", "cache_write", "attention"), "attention.reference", "reference", provider=provider)
)
