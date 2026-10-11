"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(("argmax",), "vocabulary.cuda.packed", "packed", priority=20),
    *variants(("embedding",), "vocabulary.cuda.packed_baseline", "packed-baseline"),
    *variants(("embedding",), "vocabulary.cuda.packed_baseline", "packed", priority=0),
    *variants(
        ("embedding",),
        "vocabulary.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
    *variants(
        ("embedding", "argmax"),
        "vocabulary.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
    ),
    *variants(
        ("embedding", "argmax"), "vocabulary.cuda.decode", "block-fp8", phase="decode"
    ),
    *variants(("embedding", "argmax"), "vocabulary.cuda.decode", "bf16"),
    *variants(("head_linear", "head_argmax"), "vocabulary.cuda.fused_head", "bf16"),
)
