"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


def fp8_cache(p):
    return p.get("kv_dtype") == "fp8"


IMPLEMENTATIONS = (
    *variants(
        ("attention", "attention_kv"),
        "attention.cuda.kv_fp8",
        "block-fp8",
        phase="prefill",
        priority=30,
        accepts=fp8_cache,
    ),
    *variants(
        ("attention_merge", "attention_partial", "attention_kv", "attention_q"),
        "attention.cuda.kv_fp8",
        "block-fp8",
        phase="decode",
        priority=30,
        accepts=fp8_cache,
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
