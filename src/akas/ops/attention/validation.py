"""Independent numerical and state qualification for attention."""


def pipeline_validation(kind, p, signature):
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage, encode_storage

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    values = sample_arguments(signature)
    slots, chunk, capacity = (p["slots"], p["chunk"], p["capacity"])
    rows = slots * chunk
    if "lengths" in values:
        values["lengths"][:] = chunk
        if slots > 1:
            values["lengths"][-1] = max(0, chunk - 1)
    if "positions" in values:
        values["positions"][:] = np.minimum(np.arange(slots) * 3 + 1, capacity - chunk)
    if "active" in values:
        values["active"][:] = 1
        if rows > 1:
            values["active"][-1] = 0
    rng = np.random.default_rng(47)
    for name in ("kc", "vc"):
        if name in values and values[name].dtype == np.uint8:
            values[name][:] = encode_storage(rng.normal(0, 1, values[name].shape))
    for name in ("ks", "vs"):
        if name in values:
            values[name][:] = 0.125
    if "stats" in values:
        values["stats"][..., 1] = np.abs(values["stats"][..., 1]) + 1
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "attention_chunk_active":
        expected["out"][:] = (
            np.arange(rows) % chunk < np.repeat(values["lengths"], chunk)
        ).astype("int32")
        return ValidationCase(values, expected, 0, 0)
    if kind == "attention_decode_workspace":
        for slot in range(slots):
            count = values["positions"][slot] + values["lengths"][slot]
            if not values["lengths"][slot]:
                continue
            written = min(capacity, (count + 127) // 128 * 128)
            for source, scales, out in (
                ("kc", "ks", "decoded_k"),
                ("vc", "vs", "decoded_v"),
            ):
                decoded = rounded(
                    decode_storage(values[source][slot, :, :count])
                    * np.repeat(values[scales][slot, :, :count], 128, axis=-1)
                )
                expected[out][slot, :, :count] = decoded
                expected[out][slot, :, count:written] = 0
        return ValidationCase(values, expected, 0, 0)
    if kind == "attention_chunk_hopper":
        from akas.ops.attention.validation import validation as attention_validation

        return attention_validation("attention", p, signature)
    if kind == "attention_chunk_merge":
        splits = p["splits"]
        for row in range(rows):
            for head in range(16):
                index = (
                    row // chunk,
                    head // 8,
                    slice(None),
                    row % chunk * 8 + head % 8,
                )
                if not values["active"][row]:
                    expected["out"][row, head * 256 : (head + 1) * 256] = 0
                    continue
                stats = values["stats"][index]
                factors = np.exp(stats[:, 0] - stats[:, 0].max())
                numerator = np.sum(values["partial"][index] * factors[:, None], axis=0)
                denominator = np.sum(stats[:, 1] * factors)
                gate = rounded(
                    values["projection"][row, head * 512 + 256 : head * 512 + 512]
                )
                expected["out"][row, head * 256 : (head + 1) * 256] = rounded(
                    rounded(numerator / denominator) / (1 + np.exp(-gate))
                )
        return ValidationCase(values, expected)
    splits, tile = (p["splits"], p.get("key_rows", 64))
    for slot in range(slots):
        count = int(values["positions"][slot] + values["lengths"][slot])
        tiles = ((count + tile - 1) // tile + splits - 1) // splits
        for head in range(16):
            cache_head = head // 8
            keys, vals = [values[name][slot, cache_head] for name in ("kc", "vc")]
            if keys.dtype == np.uint8:
                keys = rounded(
                    decode_storage(keys)
                    * np.repeat(values["ks"][slot, cache_head], 128, axis=-1)
                )
                vals = rounded(
                    decode_storage(vals)
                    * np.repeat(values["vs"][slot, cache_head], 128, axis=-1)
                )
            for token in range(chunk):
                limit = int(values["positions"][slot]) + token + 1
                for part in range(splits):
                    index = (slot, cache_head, part, token * 8 + head % 8)
                    acc, normalizer, maximum = (np.zeros(256, "float64"), 0.0, -np.inf)
                    if values["lengths"][slot] > 0:
                        begin, end = (
                            part * tiles * tile,
                            min(count, limit, (part + 1) * tiles * tile),
                        )
                        for start in range(begin, end, tile):
                            stop = min(start + tile, end)
                            query = (
                                values["q"][slot * chunk + token, head].astype(
                                    "float64"
                                )
                                if token < values["lengths"][slot]
                                else np.zeros(256)
                            )
                            scores = keys[start:stop].astype("float64") @ query / 16
                            next_maximum = max(maximum, scores.max())
                            correction = np.exp(maximum - next_maximum)
                            probabilities = np.exp(scores - next_maximum)
                            acc = acc * correction + rounded(probabilities).astype(
                                "float64"
                            ) @ vals[start:stop].astype("float64")
                            normalizer = normalizer * correction + probabilities.sum()
                            maximum = next_maximum
                    expected["out"][index] = acc
                    expected["stats"][index] = [maximum, normalizer]
    return ValidationCase(values, expected)


def canonical_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage, encode_storage

    values = sample_arguments(signature)
    r, d = (p["r"], p["d"])
    if kind == "rope":
        h = p["h"]
        limit = p.get("position_limit", 256)
        values["positions"][:] = (np.arange(r) * 17 + 1) % limit
        if limit > 8192:
            values["positions"][-1] = limit - 1
        expected = {
            name: value.copy() if name == "out" else value
            for name, value in values.items()
        }
        x = values["x"].reshape(r, h, d).astype("float64")
        angle = values["positions"][:, None, None] * p["theta"] ** (
            -2 * np.arange(d // 2)[None, None, :] / d
        )
        left, right = (x[:, :, : d // 2], x[:, :, d // 2 :])
        out = np.concatenate(
            (
                left * np.cos(angle) - right * np.sin(angle),
                left * np.sin(angle) + right * np.cos(angle),
            ),
            axis=-1,
        )
        expected["out"][:] = out.reshape(expected["out"].shape)
        return ValidationCase(values, expected, rtol=0.0002, atol=2e-05)
    h, kh, cap = (p["h"], p["kh"], p["capacity"])
    chunk, splits = (p.get("chunk", 1), p.get("splits", 16))
    values["enabled"][:] = 1
    if r > 1:
        values["enabled"][-1] = 0
    if "positions" in values:
        values["positions"][:] = np.minimum(
            cap - 1, np.arange(r) % chunk + max(0, cap - chunk)
        )
    quantized = p["kv_dtype"] not in ("float16", "float32")
    rng = np.random.default_rng(29)
    for name in ("kc", "vc"):
        if name not in values:
            continue
        if quantized:
            byte_shape = (*values[name].shape[:-1], d)
            total = int(np.prod(byte_shape))
            period = 1048573
            if total <= period:
                bits = rng.integers(0, 126, byte_shape, dtype="uint8")
                bits |= rng.integers(0, 2, byte_shape, dtype="uint8") << 7
            else:
                tile = rng.integers(0, 72, period, dtype="uint8")
                tile |= rng.integers(0, 2, period, dtype="uint8") << 7
                bits = np.resize(tile, byte_shape)
            values[name][:] = (
                bits.view("uint32").reshape(values[name].shape)
                if p["provider"] == "webgpu"
                else bits
            )
        elif p["kv_dtype"] == "float16":
            values[name][:] = values[name].astype("float16")
    for name in ("ks", "vs"):
        if name in values:
            values[name][:] = (
                1 if p.get("unit_scales", False) else np.abs(values[name]) + 0.01
            )
    if kind == "attention_merge":
        parts = values["parts"].reshape(r, h, splits, d + 2)
        parts[:, :, :, -1] = np.abs(parts[:, :, :, -1]) + 1
        expected = {
            name: value.copy() if name == "out" else value
            for name, value in values.items()
        }
        maxima = parts[:, :, :, -2]
        factors = np.exp(maxima - maxima.max(axis=2, keepdims=True))
        out = (
            np.sum(parts[:, :, :, :d] * factors[:, :, :, None], axis=2)
            / np.sum(parts[:, :, :, -1] * factors, axis=2)[:, :, None]
        )
        out[values["enabled"] == 0] = 0
        expected["out"][:] = out.reshape(expected["out"].shape)
        return ValidationCase(values, expected, rtol=0.0002, atol=2e-05)
    written_names = ("kc", "vc", "ks", "vs") if kind == "cache_write" else ("out",)
    expected = {
        name: value.copy() if name in written_names else value
        for name, value in values.items()
    }
    if kind == "cache_write":
        live = np.flatnonzero(values["enabled"][:r])
        if live.size:
            slots_idx = live // chunk
            pos_idx = values["positions"][live]
            blocks = (d + 127) // 128
            for source, cache, scales in (("keys", "kc", "ks"), ("values", "vc", "vs")):
                src = values[source][live].reshape(-1, kh, d)
                if quantized:
                    encoded = np.empty((live.size, kh, d), dtype="uint8")
                    for block in range(blocks):
                        b0, b1 = (block * 128, min(d, (block + 1) * 128))
                        part = src[:, :, b0:b1]
                        scale = np.maximum(
                            np.max(np.abs(part), axis=-1), np.float32(1e-12)
                        ) / np.float32(448)
                        expected[scales][slots_idx, :, pos_idx, block] = scale
                        encoded[:, :, b0:b1] = encode_storage(part / scale[:, :, None])
                    expected[cache][slots_idx, :, pos_idx] = (
                        encoded.view("uint32").reshape(live.size, kh, -1)
                        if p["provider"] == "webgpu"
                        else encoded
                    )
                else:
                    expected[cache][slots_idx, :, pos_idx] = (
                        src.astype("float16") if p["kv_dtype"] == "float16" else src
                    )
                    expected[scales][slots_idx, :, pos_idx, 0] = 1
        return ValidationCase(values, expected, rtol=0, atol=1e-09)
    decoded_cache = {}

    def read_slot(cache, scales, slot, end):
        key = (cache, slot)
        if key not in decoded_cache:
            data = values[cache][slot]
            if quantized:
                bits = (
                    data.view("uint8").reshape(kh, cap, d)
                    if p["provider"] == "webgpu"
                    else data
                )
                data = decode_storage(bits)
            factors = values[scales][slot]
            decoded_cache[key] = (
                data.astype("float64") * np.repeat(factors, 128, axis=-1)[:, :, :d]
            )
        return decoded_cache[key][:, :end]

    queries = values["q"].reshape(r, h, d).astype("float64")
    if kind == "attention_partial":
        out = expected["out"].reshape(r, h, splits, d + 2)
    elif kind == "attention":
        out = expected["out"].reshape(r, h, d)
    else:
        raise ValueError(f"attention validation missing for {kind}")
    group = h // kh
    if kind == "attention" and chunk > 1 and (p.get("prefill_precision") == "float16"):
        num_slots = r // chunk
        for slot in range(num_slots):
            r0, r1 = (slot * chunk, (slot + 1) * chunk)
            en = values["enabled"][r0:r1] != 0
            out[r0:r1][~en] = 0
            if not np.any(en):
                continue
            pos = np.minimum(cap - 1, values["positions"][r0:r1])
            max_len = int(pos[en].max()) + 1
            keys = (
                read_slot("kc", "ks", slot, max_len).astype("float16").astype("float64")
            )
            vals = (
                read_slot("vc", "vs", slot, max_len).astype("float16").astype("float64")
            )
            q_slot = (
                queries[r0:r1]
                .reshape(chunk, kh, group, d)
                .astype("float16")
                .astype("float64")
            )
            q_heads = np.ascontiguousarray(q_slot.transpose(1, 0, 2, 3)).reshape(
                kh, chunk * group, d
            )
            maximum = np.full((chunk, kh, group), -np.inf, dtype="float64")
            normalizer = np.zeros((chunk, kh, group), dtype="float64")
            accumulated = np.zeros((chunk, kh, group, d), dtype="float64")
            tile = p.get("key_rows", 64)
            for begin in range(0, max_len, tile):
                end = min(max_len, begin + tile)
                k_idx = np.arange(begin, end)
                valid = k_idx[None, :] <= pos[:, None]
                if not np.any(valid):
                    continue
                scores = (q_heads @ keys[:, begin:end].transpose(0, 2, 1)).reshape(
                    kh, chunk, group, end - begin
                ).transpose(1, 0, 2, 3) / np.sqrt(d)
                scores = np.where(valid[:, None, None, :], scores, -np.inf)
                block_max = scores.max(axis=-1)
                has_valid = valid.any(axis=-1)[:, None, None]
                next_maximum = np.where(
                    has_valid, np.maximum(maximum, block_max), maximum
                )
                correction = np.exp(maximum - next_maximum)
                probabilities = np.where(
                    valid[:, None, None, :],
                    np.exp(scores - next_maximum[:, :, :, None]),
                    0.0,
                )
                normalizer = normalizer * correction + probabilities.sum(axis=-1)
                prob_f16 = probabilities.astype("float16").astype("float64")
                prob_heads = np.ascontiguousarray(
                    prob_f16.transpose(1, 0, 2, 3)
                ).reshape(kh, chunk * group, end - begin)
                contribution = (
                    (prob_heads @ vals[:, begin:end])
                    .reshape(kh, chunk, group, d)
                    .transpose(1, 0, 2, 3)
                )
                accumulated = accumulated * correction[:, :, :, None] + contribution
                maximum = next_maximum
            res = (accumulated / normalizer[:, :, :, None]).reshape(chunk, h, d)
            out[r0:r1][en] = res[en]
        return ValidationCase(values, expected, rtol=0.02, atol=0.002)
    if kind == "attention":
        num_slots = r // chunk
        for slot in range(num_slots):
            r0, r1 = (slot * chunk, (slot + 1) * chunk)
            en = values["enabled"][r0:r1] != 0
            out[r0:r1][~en] = 0
            if not np.any(en):
                continue
            pos = np.minimum(cap - 1, values["positions"][r0:r1])
            max_len = int(pos[en].max()) + 1
            keys = read_slot("kc", "ks", slot, max_len)
            vals = read_slot("vc", "vs", slot, max_len)
            q_slot = queries[r0:r1].reshape(chunk, kh, group, d)
            q_heads = np.ascontiguousarray(q_slot.transpose(1, 0, 2, 3)).reshape(
                kh, chunk * group, d
            )
            maximum = np.full((chunk, kh, group), -np.inf, dtype="float64")
            normalizer = np.zeros((chunk, kh, group), dtype="float64")
            accumulated = np.zeros((chunk, kh, group, d), dtype="float64")
            for begin in range(0, max_len, 64):
                end = min(max_len, begin + 64)
                k_idx = np.arange(begin, end)
                valid = k_idx[None, :] <= pos[:, None]
                if not np.any(valid):
                    continue
                scores = (q_heads @ keys[:, begin:end].transpose(0, 2, 1)).reshape(
                    kh, chunk, group, end - begin
                ).transpose(1, 0, 2, 3) / np.sqrt(d)
                scores = np.where(valid[:, None, None, :], scores, -np.inf)
                block_max = scores.max(axis=-1)
                has_valid = valid.any(axis=-1)[:, None, None]
                next_maximum = np.where(
                    has_valid, np.maximum(maximum, block_max), maximum
                )
                correction = np.exp(maximum - next_maximum)
                probabilities = np.where(
                    valid[:, None, None, :],
                    np.exp(scores - next_maximum[:, :, :, None]),
                    0.0,
                )
                normalizer = normalizer * correction + probabilities.sum(axis=-1)
                prob_heads = np.ascontiguousarray(
                    probabilities.transpose(1, 0, 2, 3)
                ).reshape(kh, chunk * group, end - begin)
                contribution = (
                    (prob_heads @ vals[:, begin:end])
                    .reshape(kh, chunk, group, d)
                    .transpose(1, 0, 2, 3)
                )
                accumulated = accumulated * correction[:, :, :, None] + contribution
                maximum = next_maximum
            res = (accumulated / normalizer[:, :, :, None]).reshape(chunk, h, d)
            out[r0:r1][en] = res[en]
        return ValidationCase(values, expected, rtol=0.02, atol=0.002)
    if kind == "attention_partial" and p.get("decode_precision") == "float16":
        queries = queries.astype("float16").astype("float64")
        tile = p.get("key_rows", 64)
        for row in range(r):
            length = (
                min(cap, int(values["positions"][row]) + 1)
                if values["enabled"][row]
                else 0
            )
            keys = read_slot("kc", "ks", row, max(1, length))
            vals = read_slot("vc", "vs", row, max(1, length))
            span = ((length + tile - 1) // tile + splits - 1) // splits * tile
            for head in range(h):
                scores = keys[head // group, :length] @ queries[row, head] / np.sqrt(d)
                for part in range(splits):
                    begin, end = part * span, min(length, (part + 1) * span)
                    maximum, normalizer = -np.inf, 0.0
                    accumulated = np.zeros(d, dtype="float64")
                    for start in range(begin, end, tile):
                        stop = min(end, start + tile)
                        block = scores[start:stop]
                        next_maximum = max(maximum, block.max())
                        correction = (
                            np.exp(maximum - next_maximum)
                            if np.isfinite(maximum)
                            else 0.0
                        )
                        probabilities = np.exp(block - next_maximum)
                        accumulated = (
                            accumulated * correction
                            + probabilities.astype("float16").astype("float64")
                            @ vals[head // group, start:stop]
                        )
                        normalizer = normalizer * correction + probabilities.sum()
                        maximum = next_maximum
                    out[row, head, part, :d] = accumulated
                    out[row, head, part, d:] = [maximum, normalizer]
        return ValidationCase(values, expected, rtol=0.002, atol=0.002)
    for row in range(r):
        length = (
            min(cap, int(values["positions"][row]) + 1) if values["enabled"][row] else 0
        )
        if not length and kind == "attention":
            out[row] = 0
            continue
        slot = row // chunk
        keys_slot = read_slot("kc", "ks", slot, max(1, length))
        vals_slot = read_slot("vc", "vs", slot, max(1, length))
        for head in range(h):
            keys = keys_slot[head // group, :length]
            vals = vals_slot[head // group, :length]
            scores = keys @ queries[row, head] / np.sqrt(d)
            if kind == "attention":
                if not length:
                    out[row, head] = 0
                    continue
                probabilities = np.exp(scores - scores.max())
                out[row, head] = probabilities @ vals / probabilities.sum()
            else:
                span = (length + splits - 1) // splits
                for part in range(splits):
                    begin, end = (part * span, min(length, (part + 1) * span))
                    if begin >= end:
                        out[row, head, part, :d] = 0
                        out[row, head, part, d:] = [-np.inf, 0]
                        continue
                    maximum = scores[begin:end].max()
                    probabilities = np.exp(scores[begin:end] - maximum)
                    out[row, head, part, :d] = probabilities @ vals[begin:end]
                    out[row, head, part, d:] = [maximum, probabilities.sum()]
    return ValidationCase(values, expected, rtol=0.02, atol=0.002)


def mapped_validation(kind, p, signature, *, mapping=None):
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage, encode_storage

    f16 = any(row["dtype"] == "float16" for row in signature)

    def rounded(x):
        if f16:
            return np.asarray(x, dtype="float16").astype("float32")
        return decode_bfloat16(encode_bfloat16(x))

    values = sample_arguments(signature)
    slots, chunk = (p["slots"], p["chunk"])
    heads, kh, d, cap = (p["heads"], p["kv_heads"], p["d"], p["capacity"])
    rows = slots * chunk
    values["mapping"][:] = np.arange(slots)[::-1] if mapping is None else mapping
    starts = np.minimum(np.arange(slots) * 3, max(0, cap - chunk))
    if kind == "attention_qkv":
        values["positions"][:] = (starts[:, None] + np.arange(chunk)[None, :]).reshape(
            -1
        )
        values["active"][:] = 1
        if rows > 1:
            values["active"][-1] = 0
    else:
        values["positions"][:] = starts
        values["lengths"][:] = chunk
        if slots > 1:
            values["lengths"][-1] = max(1, chunk - 1)
    if "ks" in values:
        rng = np.random.default_rng(29)
        for name in ("kc", "vc"):
            byte_shape = (*values[name].shape[:-1], d)
            period = 1048573
            tile = rng.integers(0, 72, period, dtype="uint8")
            tile |= rng.integers(0, 2, period, dtype="uint8") << 7
            bits = np.resize(tile, byte_shape)
            values[name][:] = (
                bits.view("uint32").reshape(values[name].shape)
                if values[name].dtype == np.uint32
                else bits
            )
        for name in ("ks", "vs"):
            values[name][:] = 0.125
    written_names = (
        (("q", "kc", "vc", "ks", "vs") if "ks" in values else ("q", "kc", "vc"))
        if kind == "attention_qkv"
        else ("out",)
    )
    expected = {
        name: value.copy() if name in written_names else value
        for name, value in values.items()
    }
    if kind == "attention_qkv":
        live = np.flatnonzero(values["active"][:rows])
        freq = (p["theta"] ** (-2 * np.arange(32) / 64))[None, None, :]
        for start in range(0, live.size, 1024):
            batch_rows = live[start : start + 1024]
            proj = rounded(values["proj"][batch_rows])
            pos = values["positions"][batch_rows]
            angles = pos[:, None, None] * freq
            cos, sin = (np.cos(angles), np.sin(angles))
            q_raw = (
                proj[:, : 2 * heads * d]
                .reshape(-1, heads, 2, d)[:, :, 0, :]
                .astype("float64")
            )
            q_norm = rounded(
                q_raw
                / np.sqrt(np.mean(q_raw * q_raw, axis=-1, keepdims=True) + p["eps"])
                * (1 + values["qw"])
            )
            q_out = q_norm.copy()
            q_out[..., :32] = q_norm[..., :32] * cos - q_norm[..., 32:64] * sin
            q_out[..., 32:64] = q_norm[..., :32] * sin + q_norm[..., 32:64] * cos
            expected["q"][batch_rows] = rounded(q_out)
            k_raw = (
                proj[:, 2 * heads * d : 2 * heads * d + kh * d]
                .reshape(-1, kh, d)
                .astype("float64")
            )
            k_norm = rounded(
                k_raw
                / np.sqrt(np.mean(k_raw * k_raw, axis=-1, keepdims=True) + p["eps"])
                * (1 + values["kw"])
            )
            k_out = k_norm.copy()
            k_out[..., :32] = k_norm[..., :32] * cos - k_norm[..., 32:64] * sin
            k_out[..., 32:64] = k_norm[..., :32] * sin + k_norm[..., 32:64] * cos
            k_out = rounded(k_out)
            v_out = proj[
                :, 2 * heads * d + kh * d : 2 * heads * d + 2 * kh * d
            ].reshape(-1, kh, d)
            req = values["mapping"][batch_rows // chunk]
            for idx, (r_slot, p_pos) in enumerate(zip(req, pos)):
                if "ks" in expected:
                    for source, cache, metadata in (
                        (k_out[idx], "kc", "ks"),
                        (v_out[idx], "vc", "vs"),
                    ):
                        encoded = np.empty((kh, d), dtype="uint8")
                        for block in range(d // 128):
                            part = source[:, block * 128 : (block + 1) * 128]
                            scale = np.maximum(
                                np.max(np.abs(part), axis=-1), np.float32(1e-12)
                            ) / np.float32(448)
                            expected[metadata][r_slot, :, p_pos, block] = scale
                            encoded[:, block * 128 : (block + 1) * 128] = (
                                encode_storage(part / scale[:, None])
                            )
                        expected[cache][r_slot, :, p_pos] = (
                            encoded.view("uint32").reshape(kh, d // 4)
                            if expected[cache].dtype == np.uint32
                            else encoded
                        )
                else:
                    expected["kc"][r_slot, :, p_pos] = k_out[idx]
                    expected["vc"][r_slot, :, p_pos] = v_out[idx]
    else:
        group = heads // kh
        for batch in range(slots):
            request = values["mapping"][batch]
            length_b = int(values["lengths"][batch])
            if length_b <= 0:
                continue
            r0, r1 = (batch * chunk, batch * chunk + length_b)
            pos = np.minimum(cap - 1, starts[batch] + np.arange(length_b))
            max_len = int(pos.max()) + 1
            keys_all = values["kc"][request, :, :max_len]
            vals_all = values["vc"][request, :, :max_len]
            if keys_all.dtype in (np.uint8, np.uint32):
                k_bits = (
                    keys_all.view("uint8").reshape(kh, max_len, d)
                    if keys_all.dtype == np.uint32
                    else keys_all
                )
                v_bits = (
                    vals_all.view("uint8").reshape(kh, max_len, d)
                    if vals_all.dtype == np.uint32
                    else vals_all
                )
                keys_all = rounded(
                    decode_storage(k_bits)
                    * np.repeat(values["ks"][request, :, :max_len], 128, axis=-1)
                )
                vals_all = rounded(
                    decode_storage(v_bits)
                    * np.repeat(values["vs"][request, :, :max_len], 128, axis=-1)
                )
            keys_all, vals_all = (
                keys_all.astype("float64"),
                vals_all.astype("float64"),
            )
            q_batch = (
                values["q"][r0:r1].reshape(length_b, kh, group, d).astype("float64")
            )
            maximum = np.full((length_b, kh, group), -np.inf, dtype="float64")
            normalizer = np.zeros((length_b, kh, group), dtype="float64")
            acc = np.zeros((length_b, kh, group, d), dtype="float64")
            for start in range(0, max_len, 64):
                end = min(max_len, start + 64)
                k_idx = np.arange(start, end)
                valid = k_idx[None, :] <= pos[:, None]
                if not np.any(valid):
                    continue
                scores = np.einsum(
                    "tkgd,ksd->tkgs", q_batch, keys_all[:, start:end]
                ) / np.sqrt(d)
                scores = np.where(valid[:, None, None, :], scores, -np.inf)
                block_max = scores.max(axis=-1)
                has_valid = valid.any(axis=-1)[:, None, None]
                next_maximum = np.where(
                    has_valid, np.maximum(maximum, block_max), maximum
                )
                correction = np.exp(maximum - next_maximum)
                probability = np.where(
                    valid[:, None, None, :],
                    np.exp(scores - next_maximum[:, :, :, None]),
                    0.0,
                )
                normalizer = normalizer * correction + probability.sum(axis=-1)
                prob_bf16 = rounded(probability).astype("float64")
                acc = acc * correction[:, :, :, None] + np.einsum(
                    "tkgs,ksd->tkgd", prob_bf16, vals_all[:, start:end]
                )
                maximum = next_maximum
            attended = rounded(acc / normalizer[:, :, :, None]).reshape(
                length_b, heads, d
            )
            gates = rounded(
                values["proj"][r0:r1, : 2 * heads * d].reshape(length_b, heads, 2, d)[
                    :, :, 1, :
                ]
            )
            gated = rounded(attended / (1 + np.exp(-gates))).reshape(
                length_b, heads * d
            )
            expected["out"][r0:r1] = gated
    comparators = {}
    if kind == "attention_qkv" and "ks" in expected:

        def check_kc(actual, reference):
            act_u8 = actual.view("uint8").reshape(expected["kc"].shape[:-1] + (d,))
            ref_u8 = reference.view("uint8").reshape(expected["kc"].shape[:-1] + (d,))
            diff = np.abs(act_u8.astype("int16") - ref_u8.astype("int16"))
            mask = np.zeros(act_u8.shape[:-1], dtype=bool)
            live_rows = np.flatnonzero(values["active"][:rows])
            if live_rows.size:
                mask[
                    values["mapping"][live_rows // chunk],
                    :,
                    values["positions"][live_rows],
                ] = True
            np.testing.assert_array_equal(act_u8[~mask], ref_u8[~mask])
            if np.any(mask):
                np.testing.assert_array_less(diff[mask], 4)

        comparators["kc"] = check_kc
    if kind == "attention":
        # This callable returns a fresh tensor. Padding rows are unspecified;
        # they are not borrowed state whose previous contents must survive.
        live = (
            np.arange(rows) % chunk < np.repeat(values["lengths"], chunk)
        ) & np.repeat(values["mapping"] >= 0, chunk)

        def check_output(actual, reference):
            np.testing.assert_allclose(
                actual[live], reference[live], rtol=0.02, atol=0.002, equal_nan=False
            )

        comparators["out"] = check_output
    return ValidationCase(
        values,
        expected,
        atol=0.004 if kind == "attention_qkv" else 0.002,
        comparators=comparators,
    )


def split_validation(kind, p, signature):
    """Split projection preparation and partitioned BF16/FP8 attention."""
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage, encode_storage

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    values = sample_arguments(signature)
    rows = p.get("r", p.get("slots", 1) * p.get("chunk", 1))
    chunk = p.get("chunk", 1)
    if "active" in values:
        values["active"][:] = 1
        if rows > 1:
            values["active"][-1] = 0
    if "positions" in values:
        values["positions"][:] = np.minimum(np.arange(rows) * 7, p["capacity"] - 1)
    if "stats" in values:
        values["stats"][..., 1] = np.abs(values["stats"][..., 1]) + 1
    rng = np.random.default_rng(41)
    for name in ("kc", "vc"):
        if name in values and values[name].dtype == np.uint8:
            values[name][:] = rng.integers(32, 72, values[name].shape, dtype="uint8")
            values[name][:] |= (
                rng.integers(0, 2, values[name].shape, dtype="uint8") << 7
            )
    for name in ("ks", "vs"):
        if name in values:
            values[name][:] = 0.125
    expected = {name: value.copy() for name, value in values.items()}
    if kind in ("attention_q", "attention_kv"):
        d = values["w"].size
        query = kind == "attention_q"
        heads = values["q"].shape[1] if query else values["kc"].shape[1]
        for row in range(rows):
            for head in range(heads):
                if not values["active"][row]:
                    if query:
                        expected["q"][row, head] = 0
                    continue
                offset = head * d * (2 if query else 1)
                x = rounded(values["projection"][row, offset : offset + d]).astype(
                    "float64"
                )
                normalized = rounded(
                    x / np.sqrt(np.mean(x * x) + p["eps"]) * (1 + values["w"])
                )
                angle = values["positions"][row] * p["theta"] ** (
                    -2 * np.arange(32) / 64
                )
                out = normalized.copy()
                left, right = (normalized[:32], normalized[32:64])
                out[:32] = left * np.cos(angle) - right * np.sin(angle)
                out[32:64] = left * np.sin(angle) + right * np.cos(angle)
                out = rounded(out)
                if query:
                    expected["q"][row, head] = out
                    continue
                slot, position = (row // chunk, values["positions"][row])
                channels = rounded(values["values"][row, head * d : (head + 1) * d])
                for source, cache, metadata in (
                    (out, "kc", "ks"),
                    (channels, "vc", "vs"),
                ):
                    if values[cache].dtype == np.uint8:
                        for block in range(d // 128):
                            part = source[block * 128 : (block + 1) * 128]
                            scale = np.maximum(
                                np.max(np.abs(part)), np.float32(1e-12)
                            ) / np.float32(448)
                            expected[metadata][slot, head, position, block] = scale
                            expected[cache][
                                slot, head, position, block * 128 : (block + 1) * 128
                            ] = encode_storage(part / scale)
                    else:
                        expected[cache][slot, head, position] = source
    elif kind == "attention_partial":
        heads, d = values["q"].shape[1:]
        kh, splits = (values["kc"].shape[1], p.get("splits", 16))
        for row in range(rows):
            length = int(values["positions"][row]) + 1 if values["active"][row] else 0
            tiles = ((length + 63) // 64 + splits - 1) // splits
            for head in range(heads):
                cache_head = head // (heads // kh)
                for part in range(splits):
                    begin, end = (
                        part * tiles * 64,
                        min(length, (part + 1) * tiles * 64),
                    )
                    index = (row, cache_head, part, head % (heads // kh))
                    maximum, normalizer = (-np.inf, 0.0)
                    accumulated = np.zeros(d, dtype="float64")
                    for tile in range(begin, end, 64):
                        keys, vals = [
                            values[name][row, cache_head, tile : min(tile + 64, end)]
                            for name in ("kc", "vc")
                        ]
                        if keys.dtype == np.uint8:
                            keys = decode_storage(keys) * np.repeat(
                                values["ks"][
                                    row, cache_head, tile : min(tile + 64, end)
                                ],
                                128,
                                axis=1,
                            )
                            vals = decode_storage(vals) * np.repeat(
                                values["vs"][
                                    row, cache_head, tile : min(tile + 64, end)
                                ],
                                128,
                                axis=1,
                            )
                        scores = (
                            rounded(keys).astype("float64")
                            @ values["q"][row, head].astype("float64")
                            / np.sqrt(d)
                        )
                        next_maximum = max(maximum, scores.max())
                        correction = np.exp(maximum - next_maximum)
                        probabilities = np.exp(scores - next_maximum)
                        normalizer = normalizer * correction + probabilities.sum()
                        accumulated = accumulated * correction + rounded(
                            probabilities
                        ).astype("float64") @ rounded(vals).astype("float64")
                        maximum = next_maximum
                    expected["out"][index] = accumulated
                    expected["stats"][index] = [maximum, normalizer]
    elif kind == "attention_merge":
        d = values["partial"].shape[-1]
        for row in range(rows):
            for head in range(values["partial"].shape[1] * values["partial"].shape[3]):
                index = (row, head // 8, slice(None), head % 8)
                if not values["active"][row]:
                    expected["out"][row, head * d : (head + 1) * d] = 0
                    continue
                stats = values["stats"][index]
                factors = np.exp(stats[:, 0] - stats[:, 0].max())
                out = np.sum(
                    values["partial"][index] * factors[:, None], axis=0
                ) / np.sum(stats[:, 1] * factors)
                gate = rounded(
                    values["projection"][row, head * 2 * d + d : head * 2 * d + 2 * d]
                )
                expected["out"][row, head * d : (head + 1) * d] = rounded(
                    rounded(out) / (1 + np.exp(-gate))
                )
    else:
        raise ValueError(f"missing split attention reference: {kind}")
    return ValidationCase(values, expected)


def validation(kind, parameters, signature):
    if "heads" in parameters and any((row["name"] == "mapping" for row in signature)):
        return mapped_validation(kind, parameters, signature)
    if kind in ("attention_q", "attention_kv") or any(
        (row["name"] == "stats" for row in signature)
    ):
        return split_validation(kind, parameters, signature)
    if (
        kind == "attention"
        and "slots" in parameters
        and any((row["name"] == "projection" for row in signature))
    ):
        import numpy as np
        from akas.kernels.contracts import ValidationCase

        specs = {row["name"]: row for row in signature}
        heads, d = specs["q"]["shape"][1:]
        slots = parameters["slots"]
        inputs = [
            {**row, "name": "proj" if row["name"] == "projection" else row["name"]}
            for row in signature
        ]
        inputs.append(dict(name="mapping", shape=[slots], dtype="int32"))
        case = mapped_validation(
            kind,
            {**parameters, "heads": heads, "kv_heads": specs["kc"]["shape"][1], "d": d},
            inputs,
            mapping=np.arange(slots, dtype="int32"),
        )

        def restore(values):
            return {
                "projection" if name == "proj" else name: value
                for name, value in values.items()
                if name != "mapping"
            }

        return ValidationCase(
            restore(case.arguments), restore(case.expected), case.rtol, case.atol
        )
    return canonical_validation(kind, parameters, signature)
