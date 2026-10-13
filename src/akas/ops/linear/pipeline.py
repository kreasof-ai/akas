"""Physical projection stages for prequantized tensor-core and split-K pipelines."""


def make_kernel(kind, parameters):
    from akas.ops.linear.cuda.block_fp8 import make_kernel as projection
    from akas.ops.linear.cuda.block_fp8 import merge_kernel

    if kind == "projection_merge":
        return merge_kernel(parameters)
    if kind == "projection_partial":
        return projection("fp8_linear_mma_prequantized", parameters)
    raise ValueError("unknown projection stage")


def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.ops.linear.validation import validation as reference
    from akas.quantization.weights.block_fp8 import decode_storage

    if kind == "projection_merge":
        values = sample_arguments(signature)
        expected = {name: value.copy() for name, value in values.items()}
        output = np.zeros_like(values["out"])
        for part in range(p["partitions"]):
            output += values["partial"][:, part]
        expected["out"] = output
        return ValidationCase(values, expected, rtol=0, atol=0)
    adapted = [
        dict(spec, shape=[p["r"], p["o"]]) if spec["name"] == "out" else spec
        for spec in signature
    ]
    baseline = reference("fp8_linear", p, adapted)
    values = baseline.arguments
    if p["partitions"] == 1:
        return baseline
    x = decode_storage(values["x"]).astype("float64")
    x *= np.repeat(values["activation_scales"], 128, axis=1)
    weight = decode_storage(values["w"]).astype("float64")
    weight *= np.repeat(np.repeat(values["scales"], 128, axis=0), 128, axis=1)[
        : p["o"], : p["k"]
    ]
    shape = (p["r"], p["partitions"], p["o"])
    values["out"] = np.zeros(shape, dtype="float32")
    expected = {name: value.copy() for name, value in values.items()}
    span = p["k"] // p["partitions"]
    for part in range(p["partitions"]):
        segment = slice(part * span, (part + 1) * span)
        expected["out"][:, part] = x[:, segment] @ weight[:, segment].T
    return ValidationCase(values, expected)
