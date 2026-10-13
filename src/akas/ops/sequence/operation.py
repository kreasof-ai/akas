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
    item
    for provider in ("cuda", "webgpu")
    for item in variants(
        ("advance", "next_tokens", "chunk_controls", "chunk_last", "masked_copy"),
        "sequence.reference",
        "reference",
        provider=provider,
    )
)


IMPLEMENTATIONS += variants(("advance",), "sequence.cuda.prefill", "block-fp8", phase="commit", required=("all_tokens",))
IMPLEMENTATIONS += variants(("gather_hidden", "store_hidden"), "sequence.cuda.bf16_batch", "block-fp8")

from akas.ops.sequence.validation import VALIDATIONS

"""Row independence for graph pruning; kernel arithmetic is unchanged."""

ROW_INDEPENDENT = ("add",)

from akas.ops.sequence.tail import validation as tail_validation

IMPLEMENTATIONS += tuple(
    item for provider in ("cuda", "webgpu")
    for item in variants(("chunk_tail", "tail_controls"), "sequence.tail", "reference", provider=provider)
)
VALIDATIONS.update({name: tail_validation for name in ("chunk_tail", "tail_controls")})


# Output locations are part of the operation ABI, independent of compiler passes.
OUTPUT_ARGUMENTS = {"reference": {'chunk_tail': (-1,), 'tail_controls': (2, 3, 4), 'chunk_last': (1,), 'masked_copy': (1,), 'chunk_controls': (2, 3), 'advance': (0,), 'next_tokens': (1,)}}

from akas.ops.contracts import OutputABI

for _contract in ("bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update({
        "gather_hidden": OutputABI((-1,), ("out",)),
        "store_hidden": OutputABI((-1,), ("saved",)),
        "controls": OutputABI((2, 3), ("flat_position", "active")),
        "last_rows": OutputABI((-1,), ("out",)),
        "advance": OutputABI((0,), ("position",)),
    })
