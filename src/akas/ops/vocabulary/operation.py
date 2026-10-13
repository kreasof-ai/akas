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
    *variants(("embedding", "argmax"), "vocabulary.cuda.decode", "block-fp8"),
    *variants(("embedding", "argmax"), "vocabulary.cuda.decode", "bf16"),
    *variants(("head_linear", "head_argmax"), "vocabulary.cuda.fused_head", "bf16"),
)

IMPLEMENTATIONS += tuple(
    item
    for provider in ("cuda", "webgpu")
    for item in variants(
        ("embedding", "argmax"), "vocabulary.reference", "reference", provider=provider,
        accepts=lambda p: not p.get("ggml", False)
    )
)


from akas.ops.vocabulary.validation import VALIDATIONS


IMPLEMENTATIONS += tuple(
    item
    for provider in ("cuda", "webgpu")
    for item in variants(
        ("embedding",),
        "vocabulary.cuda.packed_baseline" if provider == "cuda" else "vocabulary.webgpu.packed",
        "reference", provider=provider, priority=40,
        accepts=lambda p: p.get("ggml", False),
    )
)


# Output locations are part of the operation ABI, independent of compiler passes.
OUTPUT_ARGUMENTS = {"reference": {'embedding': (-1,), 'argmax': (1,)}}

from akas.ops.contracts import OutputABI
OUTPUT_ARGUMENTS["reference"].update({"embedding": OutputABI((-1,), ("out",)), "argmax": OutputABI((1,), ("out",))})
