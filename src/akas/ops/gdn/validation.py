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
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "gdn_conv":
        channels = values["out"].shape[-1]
        w = values["w"][:, 0].astype("float64")
        for batch in range(slots):
            request = int(values["mapping"][batch]) if "mapping" in values else batch
            history = values["state"][request].copy()
            for token in range(chunk):
                row = batch * chunk + token
                length = (
                    values["lengths"][batch]
                    if "lengths" in values
                    else chunk
                    if values["active"][batch]
                    else 0
                )
                if token >= length:
                    expected["out"][row] = 0
                    continue
                current = rounded(values["x"][row, :channels])
                conv = np.sum(history * w[:, :3], axis=1) + current * w[:, 3]
                expected["out"][row] = rounded(conv / (1 + np.exp(-conv)))
                history[:, :2] = history[:, 1:]
                history[:, 2] = current
                if (
                    "checkpoints" in expected
                    and token < expected["checkpoints"].shape[1]
                ):
                    expected["checkpoints"][batch, token] = history
            if p.get("commit", True):
                expected["state"][request] = history
    elif kind == "restore":
        values["accepted"][:] = np.arange(slots) % (chunk + 1)
        values["lengths"][:] = chunk
        expected = {name: value.copy() for name, value in values.items()}
        for batch, count in enumerate(values["accepted"]):
            if count > 0 and (p.get("full_checkpoints", False) or count < chunk):
                expected["state"][batch] = values["checkpoints"][batch, count - 1]
    elif kind == "gdn_prepare":
        conv = values["conv"] if "conv" in values else values["qkv"]
        heads, d = values["q"].shape[1:]
        channels = conv.shape[-1]
        key_heads = (channels - heads * d) // (2 * d)
        q = conv[:, : key_heads * d].reshape(rows, key_heads, d).astype("float64")
        k = (
            conv[:, key_heads * d : 2 * key_heads * d]
            .reshape(rows, key_heads, d)
            .astype("float64")
        )
        q = q / np.sqrt(np.sum(q * q, axis=2, keepdims=True) + 1e-6) * d**-0.5
        k = k / np.sqrt(np.sum(k * k, axis=2, keepdims=True) + 1e-6)
        expected["q"][:] = np.repeat(q, heads // key_heads, axis=1)
        expected["k"][:] = np.repeat(k, heads // key_heads, axis=1)
        expected["v"][:] = conv[:, 2 * key_heads * d :].reshape(rows, heads, d)
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
        for batch in range(slots):
            request = int(values["mapping"][batch]) if "mapping" in values else batch
            matrix = values["state"][request].astype("float64")
            for token in range(chunk):
                row = batch * chunk + token
                length = (
                    values["lengths"][batch]
                    if "lengths" in values
                    else chunk
                    if values["active"][batch]
                    else 0
                )
                if token >= length:
                    expected["out"][row] = 0
                    continue
                q, k, v = [
                    values[name][row].astype("float64") for name in ("q", "k", "v")
                ]
                matrix *= np.exp(values["g" if "g" in values else "log_decay"][row])[
                    :, None, None
                ]
                prediction = np.einsum("hvk,hk->hv", matrix, k)
                delta = values["beta"][row, :, None] * (v - prediction)
                matrix += delta[:, :, None] * k[:, None, :]
                expected["out"][row] = np.einsum("hvk,hk->hv", matrix, q)
                if (
                    "checkpoints" in expected
                    and token < expected["checkpoints"].shape[1]
                ):
                    expected["checkpoints"][batch, token] = matrix
            if p.get("commit", True):
                expected["state"][request] = matrix
    elif kind == "gdn_norm":
        heads, d = values["x"].shape[1:]
        channels = p.get("channels", 0)
        x = rounded(values["x"]).astype("float64")
        gate = rounded(
            values["proj"][:, channels : channels + heads * d]
            if "proj" in values
            else values["z"]
        ).reshape(rows, heads, d)
        output = (
            x / np.sqrt(np.mean(x * x, axis=2, keepdims=True) + p["eps"]) * values["w"]
        )
        output *= gate / (1 + np.exp(-gate))
        expected["out"][:] = rounded(output.reshape(rows, heads * d))
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
