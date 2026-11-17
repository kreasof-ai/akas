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


def codebook_residual_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    groups, vocab, width = (p[key] for key in ("groups", "vocab", "c"))
    values["ids"][:] = (
        np.arange(values["ids"].size).reshape(values["ids"].shape) % vocab
    )
    table = values["table"].reshape(groups, vocab, width)
    biases = values["bias"].reshape(groups, width)
    residual = table[np.arange(groups), values["ids"]].astype("float64") + biases
    out = values["x"] + residual.sum(axis=1)
    return ValidationCase(values, {**values, "out": out.astype("float32")})


def codebook_residual_masked_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    groups, vocab, width = (p[key] for key in ("groups", "vocab", "c"))
    ids = np.arange(values["ids"].size, dtype="int32").reshape(values["ids"].shape) % vocab
    if ids.size > 1:
        ids.reshape(-1)[::3] = -1
    values["ids"][:] = ids
    table = values["table"].reshape(groups, vocab, width).astype("float64")
    biases = values["bias"].reshape(groups, width).astype("float64")
    out = values["x"].astype("float64").copy()
    for r in range(ids.shape[0]):
        for g in range(groups):
            code = int(ids[r, g])
            if 0 <= code < vocab:
                out[r] += table[g, code] + biases[g]
    return ValidationCase(values, {**values, "out": out.astype("float32")})


def zonos3_frame_embed_validation(kind, p, signature):
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    rows, width, codebooks, code_vocab, text_vocab, cond_rows = (
        p[k]
        for k in ("r", "c", "codebooks", "code_vocab", "text_vocab", "cond_rows")
    )
    rng = np.random.default_rng(71)
    audio_f = rng.standard_normal((codebooks * code_vocab, width)).astype("float32") * 0.25
    text_f = rng.standard_normal((text_vocab, width)).astype("float32") * 0.25
    values["audio_w"][:] = encode_bfloat16(audio_f).view("uint8").reshape(-1)
    values["text_w"][:] = encode_bfloat16(text_f).view("uint8").reshape(-1)
    audio_dec = decode_bfloat16(values["audio_w"].view("uint16")).reshape(
        codebooks, code_vocab, width
    )
    text_dec = decode_bfloat16(values["text_w"].view("uint16")).reshape(text_vocab, width)
    frames = rng.integers(0, min(code_vocab, 64), size=(rows, codebooks + 1), dtype="int32")
    frames[:, codebooks] = rng.integers(0, min(text_vocab, 64), size=(rows,), dtype="int32")
    if rows > 1:
        frames[0, codebooks] = -1
    if rows > 2:
        frames[-1, 0] = -1
    values["frames"][:] = frames
    out = np.zeros((rows, width), dtype="float32")
    for r in range(rows):
        if frames[r, 0] < 0:
            out[r] = 0.0
        elif frames[r, codebooks] < 0:
            out[r] = values["cond_w"][min(max(-1 - int(frames[r, codebooks]), 0), cond_rows - 1)]
        else:
            acc = text_dec[int(frames[r, codebooks])].astype("float64")
            for cb in range(codebooks):
                acc = acc + audio_dec[cb, int(frames[r, cb])].astype("float64")
            out[r] = acc.astype("float32")
    return ValidationCase(values, {**values, "out": out})


def codebook_sample_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    slots = p["slots"]
    codebooks = p.get("codebooks", 10)
    vocab = p.get("vocab", 4098)
    eoa, pad = vocab - 2, vocab - 1
    window = p.get("window", 96)
    rep_cbs = p.get("rep_codebooks", 6)
    top_k = p.get("top_k", 80)
    top_p = p.get("top_p", 0.69)
    penalty = p.get("penalty", 2.4)
    softcap = p.get("softcap", 15.0)
    silence = p.get("silence_frames", 3)
    text_pad = p.get("text_pad", 8196)

    rng = np.random.default_rng(91)
    # Distinct well-separated logits so float32 vs float64 boundary ties cannot flip discrete picks
    base_logits = rng.standard_normal((slots, codebooks, vocab)).astype("float32") * 3.0
    values["projection"][:] = base_logits.reshape(slots, codebooks * vocab)
    values["history"][:] = rng.integers(0, min(eoa, 128), size=(slots, window, codebooks), dtype="int32")
    values["steps"][:] = np.arange(slots, dtype="int32") * 37
    if slots > 1:
        values["steps"][-1] = -1
    values["temperatures"][:] = 0.75
    if slots > 2:
        values["temperatures"][0] = 0.0
    values["uniforms"][:] = np.linspace(0.11, 0.89, slots * codebooks, dtype="float32").reshape(slots, codebooks)

    expected = {k: v.copy() for k, v in values.items()}
    out = np.full((slots, codebooks + 1), pad, dtype="int32")
    out[:, codebooks] = text_pad
    for s in range(slots):
        step = int(values["steps"][s])
        if step < 0:
            continue
        temp = float(values["temperatures"][s])
        for cb in range(codebooks):
            raw = values["projection"][s, cb * vocab : (cb + 1) * vocab].astype("float64")[: eoa + 1]
            v = softcap * np.tanh(raw / softcap)
            if cb < rep_cbs and step > 0:
                used = np.unique(values["history"][s, : min(step, window), cb])
                used = used[(used >= 0) & (used < eoa)]
                v[used] = np.where(v[used] > 0, v[used] / penalty, v[used] * penalty)
            order = np.argsort(-v, kind="stable")[:top_k]
            if temp <= 0.0:
                chosen = int(order[0])
            else:
                tv = v[order] / temp
                w = np.exp(tv - tv[0])
                before = np.cumsum(w) - w
                w[before > top_p * w.sum()] = 0.0
                cdf = np.cumsum(w)
                u = float(values["uniforms"][s, cb]) * cdf[-1]
                idx = int(np.searchsorted(cdf, u, side="right"))
                chosen = int(order[min(idx, len(order) - 1)])
            tok = pad if cb > silence + step else chosen
            out[s, cb] = tok
            expected["history"][s, step % window, cb] = tok
    expected["out"] = out
    return ValidationCase(values, expected, rtol=0, atol=0)
