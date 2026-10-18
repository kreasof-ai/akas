"""Independent numerical and state qualification for ggml."""


def sample_weight(storage, kind, *, seed=31):
    """Fill resident bytes, including signed payloads and finite block scales."""
    import numpy as np
    from akas.quantization.weights.ggml.definition import ENCODINGS as TYPES
    from akas.quantization.weights.ggml.kernel import decode_host

    rng = np.random.default_rng(seed)
    if kind in (0, 1):
        if storage.size <= 1_048_576:
            storage[:] = rng.normal(0, 0.15, storage.shape)
        else:
            tile = rng.normal(0, 0.15, 262_144).astype(storage.dtype)
            storage.reshape(-1)[:] = np.resize(tile, storage.size)
        return storage.astype("float32").reshape(-1)
    raw = storage.view("uint8").reshape(-1)
    # Word storage can contain up to three tail padding bytes.
    raw = raw[: raw.size // TYPES[kind][2] * TYPES[kind][2]]
    num_blocks = raw.size // TYPES[kind][2]
    if num_blocks <= 4096:
        raw[:] = rng.integers(0, 256, raw.shape, dtype="uint8")
        blocks = raw.reshape(-1, TYPES[kind][2])
        scale_offset = 208 if kind == 14 else 0
        scales = rng.uniform(0.001, 0.004, (len(blocks), 1)).astype("float16")
        blocks[:, scale_offset : scale_offset + 2] = scales.view("uint8")
        if kind in (3, 12, 13):
            minimum = rng.uniform(0.001, 0.003, (len(blocks), 1)).astype("float16")
            blocks[:, 2:4] = minimum.view("uint8")
        return decode_host(raw, kind)
    tile_blocks = rng.integers(0, 256, (4096, TYPES[kind][2]), dtype="uint8")
    scale_offset = 208 if kind == 14 else 0
    scales = rng.uniform(0.001, 0.004, (4096, 1)).astype("float16")
    tile_blocks[:, scale_offset : scale_offset + 2] = scales.view("uint8")
    if kind in (3, 12, 13):
        minimum = rng.uniform(0.001, 0.003, (4096, 1)).astype("float16")
        tile_blocks[:, 2:4] = minimum.view("uint8")
    raw.reshape(num_blocks, TYPES[kind][2])[:] = np.resize(
        tile_blocks, (num_blocks, TYPES[kind][2])
    )
    decoded_tile = decode_host(tile_blocks.reshape(-1), kind)
    return np.resize(decoded_tile, num_blocks * TYPES[kind][1])


def validation(kind, parameters, signature):
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    decoded = sample_weight(values["w"], parameters["type"])
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = decoded.reshape(expected["out"].shape)
    return ValidationCase(values, expected, rtol=1e-6, atol=1e-6)
