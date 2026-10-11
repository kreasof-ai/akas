"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(("prefill_tail",), "sequence.cuda.packed", "packed", priority=20),
    *variants(("last", "advance"), "sequence.cuda.packed_baseline", "packed-baseline"),
    *variants(
        ("last", "advance"), "sequence.cuda.packed_baseline", "packed", priority=0
    ),
    *variants(
        ("last", "advance"),
        "sequence.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
    *variants(
        ("prefill_tail",),
        "sequence.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
    ),
    *variants(
        ("gather_hidden", "store_hidden", "controls", "last_rows"),
        "sequence.cuda.bf16_batch",
        "bf16",
    ),
    *variants(
        ("controls", "last_rows", "advance"),
        "sequence.cuda.prefill",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
    ),
)

IMPLEMENTATIONS += tuple(
    item for provider in ("cuda", "webgpu")
    for item in variants(('advance', 'next_tokens'), "sequence.reference", "reference", provider=provider)
)
