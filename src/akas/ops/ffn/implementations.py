"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(("swiglu",), "ffn.cuda.packed_baseline", "packed-baseline"),
    *variants(("swiglu",), "ffn.cuda.packed_baseline", "packed", priority=0),
    *variants(
        ("swiglu",), "ffn.cuda.packed_baseline", "packed", provider="webgpu", priority=0
    ),
    *variants(("swiglu",), "ffn.cuda.bf16_batch", "bf16"),
    *variants(
        ("swiglu", "swiglu_experts"), "ffn.cuda.decode", "block-fp8", phase="decode"
    ),
)

IMPLEMENTATIONS += tuple(
    item for provider in ("cuda", "webgpu")
    for item in variants(('swiglu',), "ffn.reference", "reference", provider=provider)
)
