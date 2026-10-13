"""Chunk attention control kernel and independent physical-stage references."""


def active(p):
    import tilelang.language as T
    slots, chunk = p["slots"], p["chunk"]

    @T.prim_func
    def kernel(lengths: T.Tensor((slots,), "int32"), out: T.Tensor((slots * chunk,), "int32")):
        with T.Kernel(T.ceildiv(slots * chunk, 64), threads=64) as block:
            row = block * 64 + T.get_thread_binding()
            if row < slots * chunk:
                out[row] = T.if_then_else(row % chunk < lengths[row // chunk], 1, 0)
    return kernel


def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.quantization.kv.fp8 import decode_storage, encode_storage
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    values = sample_arguments(signature)
    slots, chunk, capacity = p["slots"], p["chunk"], p["capacity"]
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
            values[name][:] = .125
    if "stats" in values:
        values["stats"][..., 1] = np.abs(values["stats"][..., 1]) + 1
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "attention_chunk_active":
        expected["out"][:] = (np.arange(rows) % chunk < np.repeat(values["lengths"], chunk)).astype("int32")
        return ValidationCase(values, expected, 0, 0)
    if kind == "attention_decode_workspace":
        for slot in range(slots):
            count = values["positions"][slot] + values["lengths"][slot]
            if not values["lengths"][slot]:
                continue
            written = min(capacity, ((count + 127) // 128) * 128)
            for source, scales, out in (("kc", "ks", "decoded_k"), ("vc", "vs", "decoded_v")):
                decoded = rounded(decode_storage(values[source][slot, :, :count]) * np.repeat(values[scales][slot, :, :count], 128, axis=-1))
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
                index = (row // chunk, head // 8, slice(None), row % chunk * 8 + head % 8)
                if not values["active"][row]:
                    expected["out"][row, head * 256:(head + 1) * 256] = 0
                    continue
                stats = values["stats"][index]
                factors = np.exp(stats[:, 0] - stats[:, 0].max())
                numerator = np.sum(values["partial"][index] * factors[:, None], axis=0)
                denominator = np.sum(stats[:, 1] * factors)
                gate = rounded(values["projection"][row, head * 512 + 256:head * 512 + 512])
                expected["out"][row, head * 256:(head + 1) * 256] = rounded(rounded(numerator / denominator) / (1 + np.exp(-gate)))
        return ValidationCase(values, expected)
    splits, tile = p["splits"], p.get("key_rows", 64)
    for slot in range(slots):
        count = int(values["positions"][slot] + values["lengths"][slot])
        tiles = ((count + tile - 1) // tile + splits - 1) // splits
        for head in range(16):
            cache_head = head // 8
            keys, vals = [values[name][slot, cache_head] for name in ("kc", "vc")]
            if keys.dtype == np.uint8:
                keys = rounded(decode_storage(keys) * np.repeat(values["ks"][slot, cache_head], 128, axis=-1))
                vals = rounded(decode_storage(vals) * np.repeat(values["vs"][slot, cache_head], 128, axis=-1))
            for token in range(chunk):
                limit = int(values["positions"][slot]) + token + 1
                for part in range(splits):
                    index = (slot, cache_head, part, token * 8 + head % 8)
                    acc, normalizer, maximum = np.zeros(256, "float64"), 0.0, -np.inf
                    if values["lengths"][slot] > 0:
                        begin, end = part * tiles * tile, min(count, limit, (part + 1) * tiles * tile)
                        for start in range(begin, end, tile):
                            stop = min(start + tile, end)
                            query = (values["q"][slot * chunk + token, head].astype("float64")
                                     if token < values["lengths"][slot] else np.zeros(256))
                            scores = keys[start:stop].astype("float64") @ query / 16
                            next_maximum = max(maximum, scores.max())
                            correction = np.exp(maximum - next_maximum)
                            probabilities = np.exp(scores - next_maximum)
                            acc = acc * correction + rounded(probabilities).astype("float64") @ vals[start:stop].astype("float64")
                            normalizer = normalizer * correction + probabilities.sum()
                            maximum = next_maximum
                    expected["out"][index] = acc
                    expected["stats"][index] = [maximum, normalizer]
    return ValidationCase(values, expected)
