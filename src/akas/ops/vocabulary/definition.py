"""Contracts and implementation choices for vocabulary."""

from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.ops.vocabulary.validation import VALIDATIONS as VALIDATIONS


def embed(graph, tokens, weight, out, *, rows, width, vocab):
    """Select encoded lookup or native lookup without decoding the vocabulary."""
    from akas.quantization.weights import Weight

    if isinstance(weight, Weight):
        if len(weight.parts) != 1:
            raise ValueError("embedding requires a single encoded matrix")
        graph(
            "quantized_embedding",
            dict(
                r=rows,
                c=width,
                vocab=vocab,
                output_dtype=out.dtype,
                weight_parts=weight.parts,
            ),
            tokens,
            *weight.buffers,
            out,
        )
    else:
        graph("embedding", dict(r=rows, c=width, vocab=vocab), tokens, weight, out)


OUTPUT_ARGUMENTS = {
    "*": {"quantized_embedding": OutputABI((-1,), ("out",))},
    "reference": {"embedding": (-1,), "argmax": (1,)},
}
OUTPUT_ARGUMENTS["reference"].update(
    {"embedding": OutputABI((-1,), ("out",)), "argmax": OutputABI((1,), ("out",))}
)
IMPLEMENTATIONS = (
    Implementation(
        "argmax",
        "akas.ops.vocabulary.kernel",
        "packed",
        priority=20,
        factory="argmax_packed",
        name="vocabulary.cuda.packed",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "packed-baseline",
        factory="embedding_packed",
        name="vocabulary.cuda.packed_baseline",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "packed",
        priority=0,
        factory="embedding_packed",
        name="vocabulary.cuda.packed_baseline",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        factory="embedding_packed",
        name="vocabulary.cuda.packed_baseline",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        factory="packed_word_embedding",
        name="vocabulary.webgpu.packed",
    ),
    Implementation(
        "argmax",
        "akas.ops.vocabulary.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        factory="packed_word_argmax",
        name="vocabulary.webgpu.packed",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "block-fp8",
        factory="native_embedding",
        name="vocabulary.cuda.decode",
    ),
    Implementation(
        "argmax",
        "akas.ops.vocabulary.kernel",
        "block-fp8",
        factory="native_argmax",
        name="vocabulary.cuda.decode",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "bf16",
        factory="native_embedding",
        name="vocabulary.cuda.decode",
    ),
    Implementation(
        "argmax",
        "akas.ops.vocabulary.kernel",
        "bf16",
        factory="native_argmax",
        name="vocabulary.cuda.decode",
    ),
    Implementation(
        "argmax",
        "akas.ops.vocabulary.kernel",
        "bf16",
        provider="webgpu",
        factory="native_argmax",
        name="vocabulary.cuda.decode",
    ),
    Implementation(
        "head_linear",
        "akas.ops.vocabulary.cuda",
        "bf16",
        factory="head_linear",
        name="vocabulary.cuda.fused_head",
    ),
    Implementation(
        "head_argmax",
        "akas.ops.vocabulary.cuda",
        "bf16",
        factory="head_argmax",
        name="vocabulary.cuda.fused_head",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "reference",
        accepts=lambda p: not p.get("ggml", False),
        factory="embedding",
        name="vocabulary.reference",
    ),
    Implementation(
        "argmax",
        "akas.ops.vocabulary.kernel",
        "reference",
        accepts=lambda p: not p.get("ggml", False),
        factory="argmax",
        name="vocabulary.reference",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "reference",
        provider="webgpu",
        accepts=lambda p: not p.get("ggml", False),
        factory="embedding",
        name="vocabulary.reference",
    ),
    Implementation(
        "argmax",
        "akas.ops.vocabulary.kernel",
        "reference",
        provider="webgpu",
        accepts=lambda p: not p.get("ggml", False),
        factory="argmax",
        name="vocabulary.reference",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "reference",
        priority=40,
        accepts=lambda p: p.get("ggml", False),
        factory="embedding_packed",
        name="vocabulary.cuda.packed_baseline",
    ),
    Implementation(
        "embedding",
        "akas.ops.vocabulary.kernel",
        "reference",
        provider="webgpu",
        priority=40,
        accepts=lambda p: p.get("ggml", False),
        factory="packed_word_embedding",
        name="vocabulary.webgpu.packed",
    ),
)


IMPLEMENTATIONS += tuple(
    Implementation(
        "quantized_embedding",
        f"akas.ops.vocabulary.{provider}",
        "*",
        factory="quantized_embedding",
        provider=provider,
        accepts=lambda p, provider=provider: (
            p.get("output_dtype", "bfloat16") != "bfloat16" or provider == "cuda"
        ),
    )
    for provider in ("cuda", "webgpu")
)
from .validation import quantized_embedding_validation

VALIDATIONS["quantized_embedding"] = quantized_embedding_validation


def _parallel_argmax_calls(graph):
    """Only the plain F32 argmax has this masking and output contract."""
    if graph.contract != "reference":
        return []
    return [
        call
        for call in graph.calls
        if call.kind == "argmax"
        and not call.parameters.get("ggml", False)
        and "c" in call.parameters
        and graph.buffers[
            call.arguments[0] if call.arguments else call.bindings["x"]
        ].dtype
        == "float32"
    ]


def search_policies(graph, commit, *, provider, target):
    return (
        [{"parallel_argmax": True, "block": 1024}]
        if any(call.parameters["c"] >= 4096 for call in _parallel_argmax_calls(graph))
        else []
    )


def lower(graph, commit, settings, *, provider, target):
    """Split token selection without changing feedback, ties or inactive rows."""
    from akas.graph import Call

    if settings.keys() - {"parallel_argmax", "block"}:
        raise ValueError("unknown vocabulary policy")
    enabled = settings.get("parallel_argmax", False)
    block = settings.get("block", 1024)
    if type(enabled) is not bool:
        raise ValueError("parallel argmax must be a boolean")
    if type(block) is not int or block < 1:
        raise ValueError("argmax block must be a positive integer")
    if not enabled:
        return graph, commit
    if graph.contract != "reference":
        raise ValueError("parallel argmax requires the plain reference contract")
    eligible = {id(call) for call in _parallel_argmax_calls(graph)}
    calls = []
    for index, call in enumerate(graph.calls):
        if id(call) not in eligible:
            calls.append(call)
            continue
        p = call.parameters
        names = (
            dict(zip(("x", "out", "enabled"), call.arguments))
            if call.arguments
            else call.bindings
        )
        parts = (p["c"] + block - 1) // block
        values = graph.buffer(
            f"pipeline.argmax.{index}.values", (p["r"], parts), "float32"
        )
        indices = graph.buffer(
            f"pipeline.argmax.{index}.indices", (p["r"], parts), "int32"
        )
        parameters = dict(r=p["r"], c=p["c"], block=block)
        calls.extend(
            (
                Call(
                    "argmax_partial",
                    parameters,
                    bindings=dict(
                        x=names["x"], values=values.name, indices=indices.name
                    ),
                ),
                Call(
                    "argmax_merge",
                    parameters,
                    bindings=dict(
                        values=values.name,
                        indices=indices.name,
                        enabled=names["enabled"],
                        out=names["out"],
                    ),
                    label=call.label,
                ),
            )
        )
    graph.calls = calls
    return graph, commit


OUTPUT_ARGUMENTS["reference"].update(
    {
        "argmax_partial": OutputABI((-2, -1), ("values", "indices")),
        "argmax_merge": OutputABI((-1,), ("out",)),
    }
)
IMPLEMENTATIONS += tuple(
    Implementation(
        kind,
        f"akas.ops.vocabulary.{provider}",
        "reference",
        provider=provider,
        factory=kind,
    )
    for provider in ("cuda", "webgpu")
    for kind in ("argmax_partial", "argmax_merge")
)
from .validation import parallel_argmax_validation

VALIDATIONS.update(
    {kind: parallel_argmax_validation for kind in ("argmax_partial", "argmax_merge")}
)
