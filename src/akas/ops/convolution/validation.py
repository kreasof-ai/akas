"""Independent numerical and state qualification for convolution."""


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    r, c, length, chunk = (p["r"], p["c"], p.get("length", 3), p.get("chunk", 1))
    values["enabled"][:] = chunk
    if r > 1:
        values["enabled"][-1] = max(0, chunk - 1)
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
