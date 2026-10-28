"""Independent numerical and state qualification for normalization."""


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    written_names = ("out", "residual") if kind == "add_rms" else ("out",)
    expected = {
        name: value.copy() if name in written_names else value
        for name, value in values.items()
    }
    r, c = (p["r"], p["c"])
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    out_dtype = next((row["dtype"] for row in signature if row["name"] == "out"))
    bf16 = out_dtype == "bfloat16"
    f16 = out_dtype == "float16"
    residual_dtype = next(
        (row["dtype"] for row in signature if row["name"] == "residual"), None
    )

    def rounded(x):
        if f16:
            return np.asarray(x, dtype="float16").astype("float32")
        return decode_bfloat16(encode_bfloat16(x))

    source = values.get("x", values.get("residual")).reshape(r, c)
    other = (
        (values["mixed"] if "mixed" in values else values["residual"]).reshape(r, c)
        if kind == "add_rms"
        else None
    )
    res_out = expected["residual"].reshape(r, c) if kind == "add_rms" else None
    out_2d = expected["out"].reshape(r, c)
    w_row = (1 + values["w"] if p.get("zero_centered") else values["w"]).reshape(1, c)
    for begin in range(0, r, 1024):
        end = min(r, begin + 1024)
        x = source[begin:end].astype("float64")
        if kind == "add_rms":
            if p.get("round_inputs"):
                x = rounded(x)
            x = x + other[begin:end]
            if residual_dtype == "float16":
                x = np.asarray(x, dtype="float16").astype("float64")
            elif residual_dtype == "bfloat16":
                x = decode_bfloat16(encode_bfloat16(x))
            res_out[begin:end] = x
        out = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + p["eps"])
        out *= w_row
        if bf16 or f16:
            out = rounded(out)
        out_2d[begin:end] = out
    return ValidationCase(values, expected)


VALIDATIONS = {name: validation for name in ("rms", "add_rms")}


def post_add_rms_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    x = values["x"].reshape(p["r"], p["c"]).astype("float64")
    post = (
        x / np.sqrt((x * x).mean(-1, keepdims=True) + p["eps"]) * values["post_weight"]
    ).astype("float32")
    hidden = (post + values["residual"].reshape(post.shape)).astype("float32")
    wide = hidden.astype("float64")
    normal = (
        wide / np.sqrt((wide * wide).mean(-1, keepdims=True) + p["eps"]) * values["w"]
    )
    return ValidationCase(
        values,
        {**values, "hidden": hidden.reshape(-1), "out": normal.reshape(-1)},
        rtol=0.002,
        atol=0.0002,
    )


VALIDATIONS["post_add_rms"] = post_add_rms_validation
