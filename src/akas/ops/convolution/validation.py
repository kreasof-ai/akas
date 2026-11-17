"""Independent numerical and state qualification for convolution."""


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    r, c, length, chunk = (p["r"], p["c"], p.get("length", 3), p.get("chunk", 1))
    values["enabled"][:] = chunk
    if r > 1:
        values["enabled"][-1] = max(0, chunk - 1)
    if r > 2:
        values["enabled"][1] = 0
    expected = {name: value.copy() for name, value in values.items()}
    x = values["x"].reshape(r, chunk, 3, c).astype("float64")
    w = values["w"].reshape(c, length).astype("float64")
    state = expected["state"]
    out = expected["out"].reshape(r, chunk, c)
    for row in range(r):
        for token in range(chunk):
            if token >= values["enabled"][row]:
                out[row, token] = 0
                continue
            current = x[row, token, 0] * x[row, token, 2]
            acc = current * w[:, -1] + np.sum(state[row].T * w[:, :-1], axis=1)
            out[row, token] = x[row, token, 1] * acc
            state[row, :-1] = state[row, 1:]
            state[row, -1] = current
    return ValidationCase(values, expected)


def verification_validation(kind, p, signature):
    """Independent history expectations include empty, partial and complete commits."""
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    rows, channels, length, chunk = (p[n] for n in ("r", "c", "length", "chunk"))
    expected = {n: a.copy() for n, a in values.items()}
    if kind == "conv_commit_prefix":
        values["accepted"][:] = chunk
        if rows > 1:
            values["accepted"][1] = 0
        if rows > 2:
            values["accepted"][-1] = max(0, chunk - 1)
        expected["accepted"] = values["accepted"]
        for row, n in enumerate(values["accepted"]):
            if n:
                expected["state"][row] = values["journal"][row, n - 1]
        return ValidationCase(values, expected, rtol=0, atol=0)
    if kind != "conv_verify":
        raise ValueError("Unknown convolution verification operation")
    values["enabled"][:] = chunk
    if rows > 1:
        values["enabled"][-1] = max(0, chunk - 1)
    if rows > 2:
        values["enabled"][1] = 0
    expected["enabled"] = values["enabled"]
    x = values["x"].reshape(rows, chunk, 3, channels).astype("float64")
    w = values["w"].reshape(channels, length).astype("float64")
    out = expected["out"].reshape(rows, chunk, channels)
    for row in range(rows):
        history = values["state"][row].copy()
        for token in range(chunk):
            if token < values["enabled"][row]:
                current = (x[row, token, 0] * x[row, token, 2]).astype("float32")
                acc = current.astype("float64") * w[:, -1] + np.sum(
                    history.T.astype("float64") * w[:, :-1], axis=1
                )
                out[row, token] = x[row, token, 1] * acc
                history[:-1] = history[1:]
                history[-1] = current
            else:
                out[row, token] = 0
            expected["journal"][row, token] = history
    return ValidationCase(values, expected)


def causal_depthwise_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    padded = np.pad(values["x"].astype("float64"), ((p["taps"] - 1, 0), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, p["taps"], axis=0)
    out = np.einsum("rct,ct->rc", windows, values["w"]) + values["bias"]
    return ValidationCase(values, {**values, "out": out.astype("float32")})


def causal_depthwise_state_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    slots, chunk, width, taps = (p[k] for k in ("slots", "chunk", "c", "taps"))
    values["lengths"][:] = chunk
    if slots > 1:
        values["lengths"][-1] = max(0, chunk - 1)
    if slots > 2:
        values["lengths"][0] = 0
    expected = {k: v.copy() for k, v in values.items()}
    x = values["x"].reshape(slots, chunk, width).astype("float64")
    w = values["w"].astype("float64")
    bias = values["bias"].astype("float64")
    out = np.zeros((slots, chunk, width), dtype="float32")
    for s in range(slots):
        live = int(values["lengths"][s])
        hist = values["state"][s].astype("float64").copy()
        for step in range(chunk):
            if step < live:
                cur = x[s, step]
                acc = bias + cur * w[:, -1] + np.sum(hist * w[:, :-1].T, axis=0)
                out[s, step] = acc.astype("float32")
                hist[:-1] = hist[1:]
                hist[-1] = cur
        expected["state"][s] = hist.astype("float32")
    expected["out"] = out.reshape(slots * chunk, width)
    return ValidationCase(values, expected)


def istft_stream_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {k: v.copy() for k, v in values.items()}
    if kind == "istft_polar":
        n0, n1 = p.get("n0", 736), p.get("n1", 2208)
        b0, b1 = n0 // 2 + 1, n1 // 2 + 1
        max_log = p.get("max_log_mag", 4.605170185988092)
        s = values["spectrum"].astype("float64")
        for idx, (offset, bins, name) in enumerate(((0, b0, "spec0"), (2 * b0, b1, "spec1"))):
            mag = np.exp(np.minimum(s[:, offset : offset + bins], max_log))
            ph = s[:, offset + bins : offset + 2 * bins]
            im = mag * np.sin(ph)
            im[:, 0] = 0.0
            im[:, -1] = 0.0
            expected[name] = np.concatenate((mag * np.cos(ph), im), axis=1).astype("float32")
        return ValidationCase(values, expected)

    slots, frames = p["slots"], p["frames"]
    values["lengths"][:] = frames
    if slots > 1:
        values["lengths"][-1] = max(0, frames - 1)
    expected["lengths"] = values["lengths"].copy()
    cols = frames * 16
    hop, n0, n1 = 184, 736, 2208
    t0, t1, m0_len, m1_len, fir_k = n0 - hop, n1 - hop, 862, 126, 127
    span = cols * hop
    if kind == "istft_ola":
        fr0 = values["fr0"].reshape(slots, cols, n0).astype("float64")
        fr1 = values["fr1"].reshape(slots, cols, n1).astype("float64")
        cat0 = np.zeros((slots, m0_len + span + t0), dtype="float64")
        cat1 = np.zeros((slots, m1_len + span + t1), dtype="float64")
        for s in range(slots):
            live_c = int(values["lengths"][s]) * 16
            st = values["state"][s].astype("float64")
            cat0[s, :m0_len] = st[t0 + t1 : t0 + t1 + m0_len]
            cat0[s, m0_len : m0_len + t0] = st[:t0]
            for c in range(live_c):
                cat0[s, m0_len + c * hop : m0_len + c * hop + n0] += fr0[s, c]
            cat1[s, :m1_len] = st[t0 + t1 + m0_len :]
            cat1[s, m1_len : m1_len + t1] = st[t0 : t0 + t1]
            for c in range(live_c):
                cat1[s, m1_len + c * hop : m1_len + c * hop + n1] += fr1[s, c]
        expected["cat0"] = cat0.astype("float32")
        expected["cat1"] = cat1.astype("float32")
        return ValidationCase(values, expected)
    if kind == "istft_mix_commit":
        cat0 = values["cat0"].astype("float64")
        cat1 = values["cat1"].astype("float64")
        mixer = values["mixer"].astype("float64")
        mid = np.zeros((slots, span), dtype="float32")
        for s in range(slots):
            live_span = int(values["lengths"][s]) * 16 * hop
            if live_span > 0:
                w0 = np.lib.stride_tricks.sliding_window_view(cat0[s, : live_span + fir_k - 1], fir_k)
                w1 = np.lib.stride_tricks.sliding_window_view(cat1[s, : live_span + fir_k - 1], fir_k)
                mid[s, :live_span] = (w0 @ mixer[0] + w1 @ mixer[1]).astype("float32")
                expected["state"][s, :t0] = cat0[s, m0_len + live_span : m0_len + live_span + t0]
                expected["state"][s, t0 : t0 + t1] = cat1[s, m1_len + live_span : m1_len + live_span + t1]
                expected["state"][s, t0 + t1 : t0 + t1 + m0_len] = cat0[s, live_span : live_span + m0_len]
                expected["state"][s, t0 + t1 + m0_len :] = cat1[s, live_span : live_span + m1_len]
        expected["mid"] = mid
        return ValidationCase(values, expected)
    if kind == "sinc_resample_limit":
        gain = p.get("gain", 0.70794578)
        knee = p.get("knee", 0.8912509381337456)
        filt = values["filt"].astype("float64")
        pcm = np.zeros_like(values["pcm"])
        for s in range(slots):
            live_b = int(values["lengths"][s]) * 4
            if live_b > 0:
                cat = np.concatenate((values["rs_state"][s], values["mid"][s])).astype("float64")
                win = np.lib.stride_tricks.sliding_window_view(cat, 770)[::736][:live_b]
                raw = (win @ filt.T).reshape(-1).astype("float32") * np.float32(gain)
                over = np.abs(raw) > np.float32(knee)
                if over.any():
                    w = np.float32(1.0 - knee)
                    raw[over] = np.sign(raw[over]) * (
                        np.float32(knee) + w * np.tanh((np.abs(raw[over]) - np.float32(knee)) / w)
                    )
                pcm[s, : live_b * 735] = raw
        expected["pcm"] = pcm
        return ValidationCase(values, expected)
    if kind == "sinc_resample_commit":
        for s in range(slots):
            live_span = int(values["lengths"][s]) * 4 * 736
            if live_span > 0:
                expected["rs_state"][s] = values["mid"][s, live_span - 322 : live_span]
        return ValidationCase(values, expected, rtol=0, atol=0)
    raise ValueError(f"Unknown streaming iSTFT operation: {kind}")
