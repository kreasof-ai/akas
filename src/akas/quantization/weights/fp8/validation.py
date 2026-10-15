"""Independent numerical and state qualification for fp8."""


def validation(kind, parameters, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage

    values = sample_arguments(signature)
    if kind == "dequantize_weight_fp8":
        from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

        values["scales"][:] = decode_bfloat16(encode_bfloat16(values["scales"]))
        bits = values["bits"].view("uint8")
        bits[:] = np.minimum(bits & 127, 126) | (bits & 128)
        expected = {name: value.copy() for name, value in values.items()}
        shape = tuple(
            parameters.get("shape", (parameters.get("r", 1), parameters.get("k", 128)))
        )
        scales = np.repeat(np.repeat(values["scales"], 128, axis=-2), 128, axis=-1)
        expected["out"][:] = (
            decode_storage(bits).reshape(shape) * scales[..., : shape[-2], : shape[-1]]
        ).reshape(expected["out"].shape)
        return ValidationCase(values, expected, rtol=1e-6, atol=1e-6)
    values["bits"][:] = np.arange(256, dtype="uint8")
    values["bits"][[127, 255]] = [126, 254]
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = decode_storage(values["bits"])
    return ValidationCase(values, expected, rtol=0, atol=0)
