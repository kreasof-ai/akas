"""Independent arithmetic and state qualification for gdn operations."""


def validation(kind, p, signature):
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    slots, chunk = p.get("slots", p.get("r", 1)), p.get("chunk", 1)
    rows = slots * chunk
    if "mapping" in values:
        values["mapping"][:] = np.arange(slots)[::-1]
        values["lengths"][:] = chunk
        if slots > 1:
            values["lengths"][-1] = max(0, chunk - 1)
    if "active" in values:
        values["active"][:] = 1
        if slots > 1:
            values["active"][-1] = 0
    if "lengths" in values and "mapping" not in values:
        values["lengths"][:] = chunk
        if slots > 1:
            values["lengths"][-1] = max(0, chunk - 1)
    if kind in ("gdn_scan", "gdn_recurrent"):
        decay = "g" if "g" in values else "log_decay"
        values[decay][:] = -np.abs(values[decay]) - 0.01
        values["beta"][:] = 1 / (1 + np.exp(-values["beta"]))
    written_names = {"out", "state", "checkpoints", "q", "k", "v", "g", "beta"}
    expected = {name: value.copy() if name in written_names else value for name, value in values.items()}
    if kind == "gdn_conv":
        channels = values["out"].shape[-1]
        w = values["w"][:, 0].astype("float64")
        requests = values["mapping"] if "mapping" in values else np.arange(slots)
        histories = values["state"][requests].copy()
        lengths = (
            values["lengths"]
            if "lengths" in values
            else np.where(values["active"][:slots] != 0, chunk, 0)
        )
        for token in range(chunk):
            rows_idx = np.arange(slots) * chunk + token
            live = token < lengths
            expected["out"][rows_idx[~live]] = 0
            if not np.any(live):
                continue
            active_rows = rows_idx[live]
            current = rounded(values["x"][active_rows, :channels])
            h = histories[live]
            conv = np.sum(h * w[None, :, :3], axis=2) + current * w[None, :, 3]
            expected["out"][active_rows] = rounded(conv / (1 + np.exp(-conv)))
            h[:, :, :2] = h[:, :, 1:]
            h[:, :, 2] = current
            histories[live] = h
            if "checkpoints" in expected and token < expected["checkpoints"].shape[1]:
                expected["checkpoints"][live, token] = h
        if p.get("commit", True):
            expected["state"][requests] = histories
    elif kind == "restore":
        values["accepted"][:] = np.arange(slots) % (chunk + 1)
        values["lengths"][:] = chunk
        expected = {name: value.copy() if name in written_names else value for name, value in values.items()}
        for batch, count in enumerate(values["accepted"]):
            if count > 0 and (p.get("full_checkpoints", False) or count < chunk):
                expected["state"][batch] = values["checkpoints"][batch, count - 1]
    elif kind == "gdn_prepare":
        conv = values["conv"] if "conv" in values else values["qkv"]
        heads, d = values["q"].shape[1:]
        channels = conv.shape[-1]
        key_heads = (channels - heads * d) // (2 * d)
        for begin in range(0, rows, 1024):
            end = min(rows, begin + 1024)
            n = end - begin
            q = conv[begin:end, : key_heads * d].reshape(n, key_heads, d).astype("float64")
            k = (
                conv[begin:end, key_heads * d : 2 * key_heads * d]
                .reshape(n, key_heads, d)
                .astype("float64")
            )
            q = q / np.sqrt(np.sum(q * q, axis=2, keepdims=True) + 1e-6) * d**-0.5
            k = k / np.sqrt(np.sum(k * k, axis=2, keepdims=True) + 1e-6)
            expected["q"][begin:end] = np.repeat(q, heads // key_heads, axis=1)
            expected["k"][begin:end] = np.repeat(k, heads // key_heads, axis=1)
            expected["v"][begin:end] = conv[begin:end, 2 * key_heads * d :].reshape(n, heads, d)
        start = channels + heads * d
        a = (
            rounded(
                values["proj"][:, start : start + heads]
                if "proj" in values
                else values["a"]
            )
            + values["dt"]
        )
        b = rounded(
            values["proj"][:, start + heads : start + 2 * heads]
            if "proj" in values
            else values["b"]
        )
        expected["g"][:] = -np.exp(values["A"]) * (
            np.maximum(a, 0) + np.log1p(np.exp(-np.abs(a)))
        )
        expected["beta"][:] = 1 / (1 + np.exp(-b))
    elif kind in ("gdn_scan", "gdn_recurrent"):
        requests = values["mapping"] if "mapping" in values else np.arange(slots)
        matrices = values["state"][requests].astype("float64")
        lengths = (
            values["lengths"]
            if "lengths" in values
            else np.where(values["active"][:slots] != 0, chunk, 0)
        )
        decay_name = "g" if "g" in values else "log_decay"
        for token in range(chunk):
            rows_idx = np.arange(slots) * chunk + token
            live = token < lengths
            expected["out"][rows_idx[~live]] = 0
            if not np.any(live):
                continue
            active_rows = rows_idx[live]
            q = values["q"][active_rows].astype("float64")
            k = values["k"][active_rows].astype("float64")
            v = values["v"][active_rows].astype("float64")
            g = np.exp(values[decay_name][active_rows])[:, :, None, None]
            m = matrices[live] * g
            prediction = np.einsum("bhvk,bhk->bhv", m, k)
            delta = values["beta"][active_rows, :, None] * (v - prediction)
            m += delta[:, :, :, None] * k[:, :, None, :]
            expected["out"][active_rows] = np.einsum("bhvk,bhk->bhv", m, q)
            matrices[live] = m
            if "checkpoints" in expected and token < expected["checkpoints"].shape[1]:
                expected["checkpoints"][live, token] = m
        if p.get("commit", True):
            expected["state"][requests] = matrices
    elif kind == "gdn_norm":
        heads, d = values["x"].shape[1:]
        channels = p.get("channels", 0)
        for begin in range(0, rows, 1024):
            end = min(rows, begin + 1024)
            n = end - begin
            x = rounded(values["x"][begin:end]).astype("float64")
            gate = rounded(
                values["proj"][begin:end, channels : channels + heads * d]
                if "proj" in values
                else values["z"][begin:end]
            ).reshape(n, heads, d)
            output = (
                x / np.sqrt(np.mean(x * x, axis=2, keepdims=True) + p["eps"]) * values["w"]
            )
            output *= gate / (1 + np.exp(-gate))
            expected["out"][begin:end] = rounded(output.reshape(n, heads * d))
    else:
        raise ValueError(f"missing recurrent validation recipe: {kind}")
    return ValidationCase(values, expected)


VALIDATIONS = {
    name: validation
    for name in (
        "gdn_conv",
        "gdn_prepare",
        "gdn_scan",
        "gdn_recurrent",
        "gdn_norm",
        "restore",
    )
}
from akas.ops.gdn.verification import validation as deferred_validation

VALIDATIONS.update(
    {
        name: deferred_validation
        for name in (
            "conv_commit",
            "defer_accept",
            "defer_clear",
            "gdn_deferred",
            "gdn_materialize",
        )
    }
)
