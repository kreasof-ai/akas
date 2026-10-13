"""Single-plane Q8 activation representation and independent numerical recipe."""

from akas.ops.registry import variants

IMPLEMENTATIONS = tuple(
    item for contract in ("packed", "reference")
    for item in variants(("quantize_q8",), "akas.quantization.activations.packing", contract,
                         provider="webgpu", priority=20, factory="quantize_kernel",
                         accepts=lambda p: p.get("r", 1) == 1)
)


def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    blocks = values["x"].reshape(-1, 32)
    scales = np.maximum(np.max(np.abs(blocks), axis=1) / np.float32(127), np.float32(1e-20))
    quant = np.rint(blocks / scales[:, None]).astype("int8")
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = quant.reshape(-1).view("uint32")
    expected["scales"][:] = scales
    expected["sums"][:] = quant.astype("int32").sum(axis=1, dtype="int32")
    return ValidationCase(values, expected, rtol=2e-6, atol=1e-9)


VALIDATIONS = {"quantize_q8": validation}

from akas.ops.contracts import OutputABI
OUTPUT_ARGUMENTS = {"*": {"quantize_q8": OutputABI((1, 2, 3), ("out", "scales", "sums"))}}
