"""Independent numerical and state qualification for gdn."""


def journal_validation(kind, p, signature):
    """Independent delayed-update algebra and journal bookkeeping."""
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    slots, window, pool = (p["slots"], p["window"], p["pool"])
    values["mapping"][:] = np.linspace(0, pool - 1, slots, dtype="int32")[::-1]
    if "lengths" in values:
        values["lengths"][:] = window
        if slots > 1:
            values["lengths"][-1] = max(1, window - 1)
    if "counts" in values:
        values["counts"][:] = np.arange(pool) % max(1, window)
    if "banks" in values:
        values["banks"][:] = np.arange(pool) % 2
    for name in ("g", "saved_g"):
        if name in values:
            values[name][:] = -np.abs(values[name]) - 0.01
    for name in ("beta", "saved_b"):
        if name in values:
            values[name][:] = 1 / (1 + np.exp(-values[name]))
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "defer_accept":
        for row, slot in enumerate(values["mapping"]):
            expected["counts"][slot] = max(int(values["lengths"][row]) - 1, 0)
            expected["banks"][slot] = 1 - values["banks"][slot]
    elif kind == "defer_clear":
        expected["counts"][values["mapping"]] = 0
    elif kind == "conv_commit":
        channels = p["channels"]
        for row, slot in enumerate(values["mapping"]):
            history = expected["state"][slot]
            for token in range(values["lengths"][row]):
                current = decode_bfloat16(
                    encode_bfloat16(values["x"][row * window + token, :channels])
                )
                history[:, :2] = history[:, 1:]
                history[:, 2] = current
    elif kind in ("gdn_materialize", "gdn_deferred"):
        for row, slot in enumerate(values["mapping"]):
            bank = int(values["banks"][slot])
            matrix = values["state"][slot].astype("float64")
            for token in range(values["counts"][slot]):
                matrix *= np.exp(values["saved_g"][bank, slot, token])[:, None, None]
                matrix += (
                    values["saved_v"][bank, slot, token]
                    * values["saved_b"][bank, slot, token, :, None]
                )[:, :, None] * values["saved_k"][bank, slot, token, :, None, :]
            if kind == "gdn_materialize":
                expected["state"][slot] = matrix
                continue
            for token in range(window):
                index = row * window + token
                if token >= values["lengths"][row]:
                    expected["out"][index] = 0
                    continue
                q, k, v = [
                    values[name][index].astype("float64") for name in ("q", "k", "v")
                ]
                matrix *= np.exp(values["g"][index])[:, None, None]
                predicted = np.einsum("hvk,hk->hv", matrix, k)
                correction = v - predicted
                matrix += (values["beta"][index, :, None] * correction)[:, :, None] * k[
                    :, None, :
                ]
                expected["out"][index] = np.einsum("hvk,hk->hv", matrix, q)
                if token == 0:
                    expected["state"][slot] = matrix
                else:
                    for name, source in (
                        ("saved_k", k),
                        ("saved_v", correction),
                        ("saved_g", values["g"][index]),
                        ("saved_b", values["beta"][index]),
                    ):
                        expected[name][1 - bank, slot, token - 1] = source
    else:
        raise ValueError("unknown deferred-state operation")
    return ValidationCase(values, expected)


def validation(kind, p, signature):
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    slots, chunk = (p.get("slots", p.get("r", 1)), p.get("chunk", 1))
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
    expected = {
        name: value.copy() if name in written_names else value
        for name, value in values.items()
    }
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
        expected = {
            name: value.copy() if name in written_names else value
            for name, value in values.items()
        }
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
            q = (
                conv[begin:end, : key_heads * d]
                .reshape(n, key_heads, d)
                .astype("float64")
            )
            k = (
                conv[begin:end, key_heads * d : 2 * key_heads * d]
                .reshape(n, key_heads, d)
                .astype("float64")
            )
            q = q / np.sqrt(np.sum(q * q, axis=2, keepdims=True) + 1e-06) * d ** (-0.5)
            k = k / np.sqrt(np.sum(k * k, axis=2, keepdims=True) + 1e-06)
            expected["q"][begin:end] = np.repeat(q, heads // key_heads, axis=1)
            expected["k"][begin:end] = np.repeat(k, heads // key_heads, axis=1)
            expected["v"][begin:end] = conv[begin:end, 2 * key_heads * d :].reshape(
                n, heads, d
            )
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
                x
                / np.sqrt(np.mean(x * x, axis=2, keepdims=True) + p["eps"])
                * values["w"]
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
VALIDATIONS.update(
    {
        name: journal_validation
        for name in (
            "conv_commit",
            "defer_accept",
            "defer_clear",
            "gdn_deferred",
            "gdn_materialize",
        )
    }
)
