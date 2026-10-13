"""Independent arithmetic and state qualification for linear operations."""


def validation(kind, p, signature):
    import numpy as np

    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    r, k, o = p["r"], p["k"], p["o"]
    if p.get("type", 0) not in (0, 1) and "scales" not in values:
        from akas.quantization.weights.ggml import sample_weight

        x = values["x"].reshape(r, k)
        weight = sample_weight(values["w"], p["type"]).reshape(o, k)
        # Packed tiled prefill stages both operands in half precision; decode
        # keeps F32 activations. This is an arithmetic policy, not a tolerance.
        if r > 1:
            x = x.astype("float16").astype("float32")
            weight = weight.astype("float16").astype("float32")
        out = x.astype("float64") @ weight.astype("float64").T
        if kind == "ffn":
            other = sample_weight(values["w2"], p["type"], seed=32).reshape(o, k)
            if r > 1:
                other = other.astype("float16").astype("float32")
            up = x.astype("float64") @ other.astype("float64").T
            out = out / (1 + np.exp(-out)) * up
        elif kind == "linear_add":
            out += values["residual"].reshape(r, o)
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = out.reshape(values["out"].shape)
        return ValidationCase(values, expected)
    if "scales" in values:
        from akas.quantization.weights.block_fp8 import decode_storage, encode_storage

        rng = np.random.default_rng(31)
        values["w"][:] = encode_storage(rng.normal(0, 1, values["w"].shape))
        values["scales"][:] = np.float32(0.125)
        weight = decode_storage(values["w"]).astype("float64")
        weight *= np.repeat(np.repeat(values["scales"], 128, axis=0), 128, axis=1)[
            :o, :k
        ]
        if "activation_scales" in values:
            values["x"][:] = encode_storage(rng.normal(0, 1, values["x"].shape))
            values["activation_scales"][:] = np.float32(0.125)
            x = decode_storage(values["x"]) * np.repeat(
                values["activation_scales"], 128, axis=1
            )
        else:
            blocks = values["x"].reshape(r, k // 128, 128)
            scales = np.maximum(
                np.max(np.abs(blocks), axis=2, keepdims=True), np.float32(1e-12)
            ) / np.float32(448)
            x = (decode_storage(encode_storage(blocks / scales)) * scales).reshape(r, k)
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = x.astype("float64") @ weight.T
        return ValidationCase(values, expected)
    if kind == "split_merge":
        out = np.zeros((r, o), dtype="float32")
        for part in range(p["parts"]):
            out += values["x"][:, part]
        expected["out"][:] = out
        return ValidationCase(values, expected, rtol=0, atol=0)
    if kind == "split_linear":
        block = p["bk"]
        span = ((k + block - 1) // block + p["parts"] - 1) // p["parts"] * block
        for part in range(p["parts"]):
            begin, end = part * span, min(k, (part + 1) * span)
            expected["out"][:, part] = (
                values["x"][:, begin:end].astype("float64")
                @ values["w"][:, begin:end].astype("float64").T
            )
        return ValidationCase(values, expected)
    x = values["x"].reshape(r, k).astype("float64")
    w = values["w"].reshape(o, k).astype("float64")
    out = x @ w.T
    if kind == "ffn":
        up = x @ values["w2"].reshape(o, k).astype("float64").T
        out = out / (1 + np.exp(-out)) * up
    elif kind == "linear_add":
        out += values["residual"].reshape(r, o)
    stride, offset = p.get("output_stride", o), p.get("output_offset", 0)
    expected["out"].reshape(r, stride)[:, offset : offset + o] = out
    return ValidationCase(values, expected, rtol=0.02, atol=0.002)


VALIDATIONS = {
    name: validation
    for name in (
        "linear",
        "gemv",
        "ffn",
        "linear_add",
        "bf16_linear",
        "fp8_linear",
        "split_linear",
        "split_merge",
    )
}
