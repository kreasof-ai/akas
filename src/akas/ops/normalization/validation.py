"""Independent arithmetic and state qualification for normalization operations."""


def validation(kind, p, signature):
    import numpy as np

    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    r, c = p["r"], p["c"]
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    bf16 = next(row["dtype"] for row in signature if row["name"] == "out") == "bfloat16"
    source = values.get("x", values.get("residual"))
    x = source.reshape(r, c).astype("float64")
    if kind == "add_rms":
        other = values["mixed"] if "mixed" in values else values["residual"]
        if p.get("round_inputs"):
            x = rounded(x)
        x = x + other.reshape(r, c)
        if bf16:
            x = rounded(x)
        expected["residual"][:] = x.reshape(expected["residual"].shape)
    out = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + p["eps"])
    out *= (1 + values["w"] if p.get("zero_centered") else values["w"]).reshape(1, c)
    if bf16:
        out = rounded(out)
    expected["out"][:] = out.reshape(expected["out"].shape)
    return ValidationCase(values, expected)


VALIDATIONS = {name: validation for name in ("rms", "add_rms")}
