"""Independent numerical and state qualification for ggml."""


def reference(options, shape, storage, *, seed=31):
    """Sample encoded storage and expose bounded independently decoded rows."""
    encoding, depth = options["encoding"], shape[-1]
    sample_weight(storage, encoding, seed=seed, decode=False)

    def read(start, stop):
        return decode_rows(storage, encoding, depth, start, stop)

    return read


def sample_weight(storage, kind, *, seed=31, decode=True):
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
            flat = storage.reshape(-1)
            for start in range(0, storage.size, len(tile)):
                count = min(len(tile), storage.size - start)
                flat[start : start + count] = tile[:count]
        return storage.astype("float32").reshape(-1) if decode else None
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
        return decode_host(raw, kind) if decode else None
    tile_blocks = rng.integers(0, 256, (4096, TYPES[kind][2]), dtype="uint8")
    scale_offset = 208 if kind == 14 else 0
    scales = rng.uniform(0.001, 0.004, (4096, 1)).astype("float16")
    tile_blocks[:, scale_offset : scale_offset + 2] = scales.view("uint8")
    if kind in (3, 12, 13):
        minimum = rng.uniform(0.001, 0.003, (4096, 1)).astype("float16")
        tile_blocks[:, 2:4] = minimum.view("uint8")
    destination = raw.reshape(num_blocks, TYPES[kind][2])
    for start in range(0, num_blocks, len(tile_blocks)):
        count = min(len(tile_blocks), num_blocks - start)
        destination[start : start + count] = tile_blocks[:count]
    if not decode:
        return None
    decoded_tile = decode_host(tile_blocks.reshape(-1), kind)
    return np.resize(decoded_tile, num_blocks * TYPES[kind][1])


def decode_rows(storage, kind, depth, start, stop):
    """Decode only complete logical rows, keeping validation memory tile bounded."""
    from .definition import ENCODINGS
    from .kernel import decode_host

    _, block, size = ENCODINGS[kind]
    if depth % block or not 0 <= start < stop:
        raise ValueError("invalid decoded weight row range")
    stride = depth // block * size
    raw = storage.view("uint8").reshape(-1)
    return decode_host(raw[start * stride : stop * stride], kind).reshape(
        stop - start, depth
    )


def validation(kind, parameters, signature):
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    decoded = sample_weight(values["w"], parameters["type"])
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = decoded.reshape(expected["out"].shape)
    return ValidationCase(values, expected, rtol=1e-6, atol=1e-6)
