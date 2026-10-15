"""Independent numerical and state qualification for q8."""


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    if kind == "dequantize_q8":
        packed = values["packed"].view("int8").reshape(-1, 32).astype("float32")
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = (packed * values["scales"][:, None]).reshape(
            expected["out"].shape
        )
        return ValidationCase(values, expected, rtol=1e-6, atol=1e-6)
    blocks = values["x"].reshape(-1, 32)
    blocks[0] = 0
    if len(blocks) > 1:
        blocks[1, :8] = [127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
    scales = np.maximum(
        np.max(np.abs(blocks), axis=1) / np.float32(127), np.float32(1e-20)
    )
    quant = np.rint(blocks / scales[:, None]).astype("int8")
    expected = {name: value.copy() for name, value in values.items()}
    name = "out" if "out" in expected else "packed"
    expected[name][:] = quant.reshape(-1).view("uint32")
    expected["scales"][:] = scales
    expected["sums"][:] = quant.astype("int32").sum(axis=1, dtype="int32")
    return ValidationCase(values, expected, rtol=2e-06, atol=1e-09)
