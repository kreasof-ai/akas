"""Independent numerical and state qualification for fp8."""


def validation(kind, parameters, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import encode_storage, decode_storage

    values = sample_arguments(signature)
    if kind == "dequantize_fp8":
        bits = values["bits"].view("uint8")
        bits[:] = np.minimum(bits & 127, 126) | (bits & 128)
        scales = values["scales"]
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = (
            decode_storage(bits).reshape(*scales.shape, 128) * scales[..., None]
        ).reshape(expected["out"].shape)
        return ValidationCase(values, expected, rtol=1e-6, atol=1e-6)
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

    stress = values["x"].reshape(-1, 128)
    stress[0] = 0
    if len(stress) > 1:
        stress[1, :8] = [448, -448, 1.0625, 1.1875, 2**-10, -(2**-10), 0, -0.0]
    values["x"][:] = decode_bfloat16(encode_bfloat16(values["x"]))
    depth = parameters["k"]
    blocks = values["x"].reshape(*values["x"].shape[:-1], depth // 128, 128)
    scales = np.maximum(
        np.max(np.abs(blocks), axis=-1), np.float32(1e-12)
    ) / np.float32(448)
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"].view("uint8")[:] = encode_storage(
        blocks / scales[..., None]
    ).reshape(values["out"].view("uint8").shape)
    expected["scales"][:] = scales
    return ValidationCase(values, expected, rtol=0, atol=1e-09)
