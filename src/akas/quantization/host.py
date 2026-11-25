"""Host storage conversion and physical layout helpers."""


def pack_words(data):
    """Preserve bytes and zero-pad the final physical WebGPU storage word."""
    import numpy as np

    raw = np.ascontiguousarray(data, dtype="uint8").reshape(-1)
    if raw.size % 4:
        raw = np.pad(raw, (0, -raw.size % 4))
    return raw.view("uint32")


def decode_storage(bits):
    import numpy as np

    bits = np.asarray(bits, dtype="uint8")
    exponent = (bits >> 3 & 15).astype("int32")
    fraction = (bits & 7).astype("float32")
    magnitude = np.where(
        exponent == 0, fraction / 512, (1 + fraction / 8) * np.exp2(exponent - 7)
    ).astype("float32")
    return np.where(bits & 128, -magnitude, magnitude).astype("float32")


def encode_storage(values):
    """Nearest-even E4M3 with finite saturation after explicit F32 rounding."""
    import numpy as np

    values = np.asarray(values, dtype="float32")
    bits = values.view("uint32")
    magnitude = np.abs(values)
    rounded = (bits & np.uint32(2147483647)) + np.uint32(524287) + (bits >> 20 & 1)
    normal = (rounded >> 20).astype("int32") - 960
    subnormal = np.rint(magnitude * np.float32(512)).astype("int32")
    code = np.clip(
        np.where(magnitude < np.float32(0.015625), subnormal, normal), 0, 126
    )
    return code.astype("uint8") | (bits >> 24 & 128).astype("uint8")


def restore_order(values, axis, order):
    """Host counterpart of grouped-to-tiled addressing for checkpoint inspection."""
    import numpy as np

    if order is None:
        return values
    start, groups, repeats, width = order
    positions = np.arange(values.shape[axis])
    tail = positions[start:] - start
    positions[start:] = (
        start
        + (tail // width % repeats * groups + tail // (width * repeats)) * width
        + tail % width
    )
    return np.take(values, positions, axis=axis)
