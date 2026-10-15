"""Contracts and implementation choices for ggml."""

from akas.quantization.formats import Format
from akas.kernels.contracts import Implementation, OutputABI
from .validation import validation

# Encoding ID: (name, logical values per block, encoded bytes per block).
ENCODINGS = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
}


def weight_storage(kind, elements, *, packed_words=False):
    if kind not in ENCODINGS:
        raise ValueError(f"unsupported GGML encoding: {kind}")
    _, block, size = ENCODINGS[kind]
    if type(elements) is not int or elements <= 0 or elements % block:
        raise ValueError(
            f"GGML {ENCODINGS[kind][0]} requires whole blocks of {block} values"
        )
    if kind in (0, 1):
        return (elements, "float16" if kind == 1 else "float32")
    count = elements // block * size
    if packed_words:
        return ((count + 3) // 4, "uint32")
    return (count, "uint8")


def layout(shape, *, encoding):
    from math import prod
    from akas.memory.state import BufferSpec
    from akas.quantization.formats import Storage

    weight_storage(encoding, shape[-1])
    count, dtype = weight_storage(encoding, prod(shape))
    return Storage(BufferSpec((count,), dtype))


def webgpu_layout(shape, *, encoding):
    from math import prod
    from akas.memory.state import BufferSpec
    from akas.quantization.formats import Storage

    weight_storage(encoding, shape[-1])
    count, dtype = weight_storage(encoding, prod(shape), packed_words=True)
    return Storage(BufferSpec((count,), dtype))


def integer_layout(shape, *, prepacked=False):
    """Derived Q4_0 word cache used by packed integer consumers."""
    if not prepacked:
        return webgpu_layout(shape, encoding=2)
    from math import prod
    from akas.memory.state import BufferSpec
    from akas.quantization.formats import Storage

    weight_storage(2, shape[-1])
    return Storage(BufferSpec((prod(shape) // 32 * 9,), "uint32"))


FORMAT = Format(
    "ggml",
    layout,
    (
        ("cuda", "akas.quantization.weights.ggml.cuda"),
        ("webgpu", "akas.quantization.weights.ggml.webgpu"),
    ),
    layouts=(("webgpu", webgpu_layout),),
)


OUTPUT_ARGUMENTS = {
    "*": {
        "dequantize_ggml": OutputABI((-1,), ("out",)),
    },
}

VALIDATIONS = {kind: validation for kind in ("dequantize_ggml",)}
IMPLEMENTATIONS = tuple(
    Implementation(
        "dequantize_ggml",
        f"akas.quantization.weights.ggml.{provider}",
        "*",
        factory="dequantize",
        provider=provider,
    )
    for provider in ("cuda", "webgpu")
)
