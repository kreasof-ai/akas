"""Independent numerical and state qualification for mtp."""


def validation(kind, parameters, signature):
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments

    f16 = any(row["dtype"] == "float16" for row in signature)

    def rounded(values):
        if f16:
            return np.asarray(values, dtype="float16").astype("float32")
        return decode_bfloat16(encode_bfloat16(values))

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "mtp_cast":
        expected["out"][:] = rounded(values["x"])
        return ValidationCase(values, expected, rtol=0, atol=0)
    width = parameters["c"]
    for index, name in enumerate(("embedding", "hidden")):
        source = values[name].astype("float64")
        out = source / np.sqrt(
            np.mean(source * source, axis=1, keepdims=True) + parameters["eps"]
        )
        out *= 1 + values[name + "_weight"]
        expected["out"][:, index * width : (index + 1) * width] = rounded(out)
    return ValidationCase(values, expected)
