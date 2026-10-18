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
