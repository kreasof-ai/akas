"""Independent numerical and state qualification for vocabulary."""


def quantized_embedding_validation(kind, p, signature):
    import numpy as np
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.ggml.validation import sample_weight

    values = sample_arguments(signature)
    decoded = sample_weight(
        values["w"], p["weight_parts"][0]["options"]["encoding"]
    ).reshape(p["vocab"], p["c"])
    values["ids"][:] = np.arange(p["r"]) % p["vocab"]
    if p["r"] >= 3:
        values["ids"][0], values["ids"][-1] = -1, p["vocab"]
    expected = {
        name: value.copy() if name == "out" else value for name, value in values.items()
    }
    selected = decoded[np.clip(values["ids"], 0, p["vocab"] - 1)].copy()
    selected[(values["ids"] < 0) | (values["ids"] >= p["vocab"])] = 0
    expected["out"][:] = (
        decode_bfloat16(encode_bfloat16(selected))
        if p.get("output_dtype", "bfloat16") == "bfloat16"
        else selected
    )
    return ValidationCase(values, expected, rtol=0, atol=0)


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    r, c = (p["r"], p.get("c", p.get("vocab")))
    if kind == "head_linear":
        logits = values["x"].astype("float64") @ values["w"].astype("float64").T
        expected = {name: value.copy() for name, value in values.items()}
        for tile in range(values["maximum"].shape[1]):
            begin, end = (tile * p["n"], min((tile + 1) * p["n"], p["o"]))
            section = logits[:, begin:end]
            expected["maximum"][:, tile] = np.max(section, axis=1)
            expected["index"][:, tile] = begin + np.argmax(section, axis=1)
        return ValidationCase(values, expected)
    if kind == "head_argmax":
        tiles, vocab = (p["tiles"], p["vocab"])
        values["index"][:] = np.arange(tiles)[None, :] * ((vocab + tiles - 1) // tiles)
        values["active"][:] = 1
        if r > 1:
            values["active"][-1] = 0
        if tiles > 1:
            values["maximum"][0, :2] = 1
            values["index"][0, :2] = [7, 3]
        expected = {name: value.copy() for name, value in values.items()}
        best = np.max(values["maximum"], axis=1, keepdims=True)
        winners = np.min(
            np.where(values["maximum"] == best, values["index"], vocab), axis=1
        )
        expected["predicted"][:] = np.where(values["active"] != 0, winners, -1)
        return ValidationCase(values, expected, rtol=0, atol=0)
    if kind == "embedding":
        v, start = (p.get("v", p.get("vocab")), p.get("start", 0))
        token_name = "tokens" if "tokens" in values else "ids"
        values[token_name][:] = start + np.arange(r) % v
        expected = {
            name: value.copy() if name == "out" else value
            for name, value in values.items()
        }
        if p.get("type", 0) not in (0, 1):
            from akas.quantization.weights.ggml.validation import (
                sample_weight,
                decode_rows,
            )

            sample_weight(values["w"], p["type"], decode=False)
            selected = np.empty((r, c), dtype="float32")
            for row, token in enumerate(values[token_name].reshape(-1)):
                selected[row] = decode_rows(
                    values["w"],
                    p["type"],
                    c,
                    int(token) - start,
                    int(token) - start + 1,
                )[0]
        else:
            weight = values["w"].reshape(v, c)
            selected = weight[values[token_name] - start]
        expected["out"][:] = selected.reshape(expected["out"].shape)
    else:
        active_name = "enabled" if "enabled" in values else "active"
        output_name = "out" if "out" in values else "tokens"
        x_name = "x" if "x" in values else "logits"
        values[active_name][:] = 1
        if r > 1:
            values[active_name][-1] = 0
        expected = {name: value.copy() for name, value in values.items()}
        winners = np.argmax(values[x_name].reshape(r, c), axis=1)
        active = values[active_name] != 0
        expected[output_name][active] = winners[active]
        if "positions" in values:
            expected[output_name][~active] = -1
            expected["positions"][active] += 1
    return ValidationCase(values, expected, rtol=0, atol=0)


VALIDATIONS = {
    name: validation for name in ("embedding", "argmax", "head_linear", "head_argmax")
}


def parallel_argmax_validation(kind, p, signature):
    """Independent first-index reduction; NaNs and -inf retain the sentinel."""
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    rows, columns, block = p["r"], p["c"], p["block"]
    parts = (columns + block - 1) // block
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "argmax_partial":
        x = values["x"].reshape(rows, columns)
        for part in range(parts):
            begin, end = part * block, min(columns, (part + 1) * block)
            section = np.where(np.isnan(x[:, begin:end]), -np.inf, x[:, begin:end])
            peaks = np.max(section, axis=1)
            expected["values"][:, part] = peaks
            expected["indices"][:, part] = np.where(
                peaks > -np.inf, begin + np.argmax(section, axis=1), columns
            )
    else:
        values["enabled"][:] = 1
        if rows > 1:
            values["enabled"][-1] = 0
        values["indices"][:] = np.arange(parts)[None, :] * block
        expected = {name: value.copy() for name, value in values.items()}
        section = np.where(np.isnan(values["values"]), -np.inf, values["values"])
        peaks = np.max(section, axis=1, keepdims=True)
        indices = np.min(np.where(section == peaks, values["indices"], columns), axis=1)
        indices = np.where(peaks[:, 0] > -np.inf, indices, columns)
        active = values["enabled"] != 0
        expected["out"][active] = indices[active]
    return ValidationCase(values, expected, rtol=0, atol=0)
