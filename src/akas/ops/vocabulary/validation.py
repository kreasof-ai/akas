"""Independent arithmetic and state qualification for vocabulary operations."""


def validation(kind, p, signature):
    import numpy as np

    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    r, c = p["r"], p.get("c", p.get("vocab"))
    if kind == "head_linear":
        logits = values["x"].astype("float64") @ values["w"].astype("float64").T
        expected = {name: value.copy() for name, value in values.items()}
        for tile in range(values["maximum"].shape[1]):
            begin, end = tile * p["n"], min((tile + 1) * p["n"], p["o"])
            section = logits[:, begin:end]
            expected["maximum"][:, tile] = np.max(section, axis=1)
            expected["index"][:, tile] = begin + np.argmax(section, axis=1)
        return ValidationCase(values, expected)
    if kind == "head_argmax":
        tiles, vocab = p["tiles"], p["vocab"]
        values["index"][:] = np.arange(tiles)[None, :] * ((vocab + tiles - 1) // tiles)
        values["active"][:] = 1
        if r > 1:
            values["active"][-1] = 0
        # Tied maxima must select the smallest vocabulary index, regardless
        # of which physical tile contains it.
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
        v, start = p.get("v", p.get("vocab")), p.get("start", 0)
        if p.get("type", 0) not in (0, 1):
            from akas.quantization.weights.ggml import sample_weight

            weight = sample_weight(values["w"], p["type"]).reshape(v, c)
        else:
            weight = values["w"].reshape(v, c)
        token_name = "tokens" if "tokens" in values else "ids"
        values[token_name][:] = start + np.arange(r) % v
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = weight[values[token_name] - start].reshape(
            expected["out"].shape
        )
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
