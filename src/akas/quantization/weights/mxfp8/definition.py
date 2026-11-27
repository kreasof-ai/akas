"""MXFP8 expert banks with row-expanded E8M0 block scales."""

from akas.memory.state import BufferSpec
from akas.kernels.spec import Implementation, OutputABI
from .validation import validation
from akas.quantization.formats import Format, Storage


def layout(shape):
    if len(shape) != 3 or any(size % 32 for size in shape[-2:]):
        raise ValueError("MXFP8 expert weights require complete 32x32 blocks")
    experts, rows, depth = shape
    return Storage(
        BufferSpec(shape, "uint8"),
        (BufferSpec((experts, rows, depth // 32), "uint8"),),
        ("scale",),
    )


FORMAT = Format("mxfp8", layout, (("cuda", "akas.quantization.weights.mxfp8.cuda"),))
OUTPUT_ARGUMENTS = {"*": {"dequantize_weight_mxfp8": OutputABI((-1,), ("out",))}}
VALIDATIONS = {"dequantize_weight_mxfp8": validation}
IMPLEMENTATIONS = (
    Implementation(
        "dequantize_weight_mxfp8",
        "akas.quantization.weights.mxfp8.cuda",
        "*",
        factory="dequantize",
    ),
)
