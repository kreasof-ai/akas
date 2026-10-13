"""Independent arithmetic and state qualification for sequence operations."""


def validation(kind, p, signature):
    import numpy as np

    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    r, chunk = p.get("r", p.get("slots", 1)), p.get("chunk", 1)
    if "enabled" in values:
        values["enabled"][:] = chunk
        if r > 1:
            values["enabled"][-1] = 0
    if "lengths" in values:
        values["lengths"][:] = chunk
        if r > 1:
            values["lengths"][-1] = max(0, chunk - 1)
    if "positions" in values:
        values["positions"][:] = np.arange(r)
    if "position" in values:
        values["position"][:] = np.arange(r)
    if "mapping" in values:
        pool = values["saved"].shape[0]
        values["mapping"][:] = np.linspace(0, pool - 1, r, dtype="int32")[::-1]
    if "mode" in values:
        values["mode"][:] = 1
    if kind == "next_tokens":
        values["sampled"][:] = np.arange(r) + 31
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "gather_hidden":
        for row, slot in enumerate(values["mapping"]):
            expected["out"][row * chunk] = values["saved"][slot]
    elif kind == "store_hidden":
        for row, slot in enumerate(values["mapping"]):
            if values["lengths"][row] > 0:
                expected["saved"][slot] = values["x"][row]
    elif kind == "controls":
        expected["flat_position"][:] = (
            values["position"][:, None] + np.arange(chunk)[None, :]
        ).reshape(-1)
        expected["active"][:] = (
            np.arange(chunk)[None, :] < values["lengths"][:, None]
        ).reshape(-1)
    elif kind == "last_rows":
        x = values["x"].reshape(r, chunk, p.get("c", p.get("width")))
        out = x[np.arange(r), np.maximum(values["lengths"] - (0 if p.get("all_tokens", False) else 1), 0)].copy()
        out[values["lengths"] == 0] = 0
        expected["out"][:] = out.reshape(expected["out"].shape)
    elif kind == "advance":
        if "position" in values:
            expected["position"] += np.maximum(values["lengths"] - (0 if p.get("all_tokens", False) else 1), 0)
            expected["active"][:] = values["lengths"] > 0
        else:
            expected["positions"] += np.where(
                values["enabled"] != 0, values["enabled"] if chunk > 1 else 1, 0
            )
    elif kind == "next_tokens":
        expected["tokens"][:] = values["sampled"]
    elif kind == "masked_copy":
        c = p["c"]
        expected["out"].reshape(r, c)[values["enabled"] != 0] = values["x"].reshape(
            r, c
        )[values["enabled"] != 0]
    elif kind == "chunk_controls":
        lengths = values["lengths"]
        if lengths.shape != (r,):
            raise ValueError("invalid chunk lengths")
        expected["row_positions"][:] = (
            values["positions"][:, None]
            + np.minimum(np.arange(chunk)[None, :], np.maximum(lengths[:, None] - 1, 0))
        ).reshape(-1)
        expected["enabled"][:] = (np.arange(chunk)[None, :] < lengths[:, None]).reshape(
            -1
        )
    elif kind == "chunk_last":
        x = values["x"].reshape(r, chunk, p["c"])
        expected["out"][:] = x[
            np.arange(r), np.maximum(values["lengths"] - (0 if p.get("all_tokens", False) else 1), 0)
        ].reshape(expected["out"].shape)
    else:
        raise ValueError(f"missing control validation for {kind}")
    return ValidationCase(values, expected, rtol=0, atol=0)


VALIDATIONS = {
    name: validation
    for name in (
        "gather_hidden",
        "store_hidden",
        "controls",
        "last_rows",
        "advance",
        "next_tokens",
        "masked_copy",
        "chunk_controls",
        "chunk_last",
    )
}
