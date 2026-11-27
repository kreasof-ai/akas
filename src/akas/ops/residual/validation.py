"""Independent numerical and state qualification for residual."""


def validation(kind, p, signature):
    from akas.validation import ValidationCase, sample_arguments

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


def bank_validation(kind, p, signature):
    import numpy as np

    from akas.validation import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    index = p["index"]
    bank = values["bank"].copy()
    bank[index] = values["source"]
    prefix = bank[: index + 1].astype("float64")
    inverse_rms = 1 / np.sqrt(np.mean(prefix**2, axis=-1) + p["eps"])
    scores = (prefix @ values["query"].astype("float64")) * inverse_rms
    weights = np.exp(scores - np.max(scores, axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    out = np.einsum("ir,irc->rc", weights, prefix).astype("float32")
    return ValidationCase(values, {**values, "bank": bank, "out": out})


def scaled_residual_validation(kind, p, signature):
    from akas.validation import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    out = (
        values["residual"].astype("float64")
        + values["mixed"].astype("float64") * values["scale"]
    )
    return ValidationCase(values, {**values, "out": out.astype("float32")})


def block_mix_validation(kind, p, signature):
    import numpy as np
    from akas.validation import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    if kind == "message_blend":
        out = (1 - np.tanh(values["forget"])) * values["x"] + np.tanh(
            values["gain"]
        ) * values["message"]
    else:
        blocks = values["x"].astype("float64")
        normalized = blocks / np.sqrt(
            np.mean(blocks**2, axis=-1, keepdims=True) + p["eps"]
        )
        scores = (
            normalized @ values["query"][p["index"]].astype("float64") / np.sqrt(p["c"])
        )
        gates = np.exp(scores - scores.max(axis=-1, keepdims=True))
        gates /= gates.sum(axis=-1, keepdims=True)
        out = np.einsum("rb,rbc->rc", gates, normalized)
    return ValidationCase(values, {**values, "out": out.astype("float32")})
