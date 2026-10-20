"""Independent pair conversion expectations, including unchanged inactive slots."""


def validation(kind, parameters, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import encode_storage, decode_storage
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

    if kind == "kv_stage":
        return stage_validation(parameters, signature)
    p = parameters
    slots, heads, depth, capacity = (
        p[n] for n in ("slots", "kv_heads", "d", "capacity")
    )
    chunk = p.get("chunk", 1)
    rows, blocks = slots * chunk, (depth + 127) // 128
    values = sample_arguments(signature)
    if kind == "kv_encode":
        values["positions"][:] = np.arange(rows, dtype="int32") % capacity
        values["active"][:] = np.arange(rows) % 3 != 1
        values["keys"][0, : min(depth, 128)] = 0
        values["values"][0, : min(depth, 128)] = 0
        values["keys"][:] = decode_bfloat16(encode_bfloat16(values["keys"]))
    else:
        for name in ("kc", "vc"):
            bits = values[name].view("uint8")
            bits[:] = np.minimum(bits & 127, 126) | (bits & 128)
    expected = {name: value.copy() for name, value in values.items()}
    logical = (slots, heads, capacity, depth)
    if kind == "kv_decode":
        for cache, metadata, output in (("kc", "ks", "keys"), ("vc", "vs", "values")):
            bits = (
                values[cache]
                .view("uint8")
                .reshape(slots, heads, capacity, -1)[..., :depth]
            )
            factors = np.repeat(values[metadata], 128, axis=-1)[..., :depth]
            result = decode_storage(bits) * factors
            expected[output][:] = decode_bfloat16(encode_bfloat16(result))
    else:
        for source, cache, metadata in (("keys", "kc", "ks"), ("values", "vc", "vs")):
            prepared = decode_bfloat16(encode_bfloat16(values[source])).reshape(
                rows, heads, depth
            )
            for row in range(rows):
                if not values["active"][row]:
                    continue
                slot, position = row // chunk, int(values["positions"][row])
                for block in range(blocks):
                    start, end = block * 128, min((block + 1) * 128, depth)
                    tile = prepared[row, :, start:end]
                    scales = np.maximum(
                        np.abs(tile).max(-1), np.float32(1e-12)
                    ) / np.float32(448)
                    expected[metadata][slot, :, position, block] = scales
                    bits = encode_storage(tile / scales[:, None])
                    target = (
                        expected[cache]
                        .view("uint8")
                        .reshape(slots, heads, capacity, -1)
                    )
                    target[slot, :, position, start:end] = bits
    return ValidationCase(values, expected, rtol=0, atol=1e-9)


def stage_validation(p, signature):
    """Independent bounded conversion, including poisoned unused cache rows."""
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage

    values = sample_arguments(signature)
    slots, heads, capacity, depth = (
        p[name] for name in ("slots", "kv_heads", "capacity", "d")
    )
    values["lengths"][:] = np.linspace(0, capacity, slots + 1, dtype="int32")[1:]
    if slots > 1:
        values["lengths"][0] = 0
    expected = dict(values)
    for name in ("keys", "values"):
        expected[name] = np.zeros_like(values[name])
    expected["scales"] = np.ones_like(values["scales"])
    for slot in range(slots):
        end = int(values["lengths"][slot])
        for source, metadata, target in (("kc", "ks", "keys"), ("vc", "vs", "values")):
            bits = values[source].view("uint8").reshape(slots, heads, capacity, -1)
            bits[slot, :, end:] = 127
            values[metadata][slot, :, end:] = np.nan
            for start in range(0, end, 128):
                stop = min(start + 128, end)
                block = decode_storage(bits[slot, :, start:stop, :depth])
                scales = np.repeat(values[metadata][slot, :, start:stop], 128, axis=-1)[
                    ..., :depth
                ]
                expected[target][slot, :, start:stop] = (block * scales).astype(
                    "float16"
                )
    return ValidationCase(values, expected, rtol=0, atol=0)
