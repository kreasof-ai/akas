"""Independent numerical and state qualification for residual."""


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    result = values["x"].astype("float32") + values["y"].astype("float32")
    dtype = next(row["dtype"] for row in signature if row["name"] == "out")
    if dtype == "bfloat16":
        from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

        result = decode_bfloat16(encode_bfloat16(result))
    elif dtype == "float16":
        result = result.astype("float16")
    expected["out"][:] = result
    return ValidationCase(values, expected, rtol=1e-06, atol=1e-07)
