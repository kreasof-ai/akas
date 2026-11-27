"""NVFP4 expert banks with row-local E4M3 and per-expert outer scales."""

from akas.memory.state import BufferSpec
from akas.kernels.spec import Implementation, OutputABI
from .validation import validation
from akas.quantization.formats import Format, Storage


def layout(shape):
    if len(shape) != 3 or shape[-1] % 16:
        raise ValueError("NVFP4 expert weights require complete 16-value blocks")
    experts, rows, depth = shape
    return Storage(
        BufferSpec((experts, rows, depth // 2), "uint8"),
        (
            BufferSpec((experts, rows, depth // 16), "uint8"),
            BufferSpec((experts,), "float32"),
            BufferSpec((1,), "float32"),
        ),
        ("scale", "amax", "denominator"),
    )


FORMAT = Format("nvfp4", layout, (("cuda", "akas.quantization.weights.nvfp4.cuda"),))
OUTPUT_ARGUMENTS = {"*": {"dequantize_weight_nvfp4": OutputABI((-1,), ("out",))}}
VALIDATIONS = {"dequantize_weight_nvfp4": validation}
IMPLEMENTATIONS = (
    Implementation(
        "dequantize_weight_nvfp4",
        "akas.quantization.weights.nvfp4.cuda",
        "*",
        factory="dequantize",
    ),
)
