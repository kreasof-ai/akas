"""Independent host reconstruction of NVFP4 weights."""


def decode_rows(values, scales, amax, denominator):
    import numpy as np
    from akas.quantization.host import decode_storage

    levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype="float32")
    codes = np.stack((values & 15, values >> 4), axis=-1)
    coefficients = np.where(codes < 8, levels[codes & 7], -levels[codes & 7])
    coefficients = coefficients.reshape(*values.shape[:-1], values.shape[-1] * 2)
    weights = coefficients * np.repeat(decode_storage(scales), 16, axis=-1)
    return weights * (amax / denominator)


def validation(kind, p, signature):
    import numpy as np
    from akas.validation import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    rng = np.random.default_rng(73)
    values["w"][:] = rng.integers(0, 256, values["w"].shape, dtype="uint8")
    values["scale"][:] = rng.integers(0, 127, values["scale"].shape, dtype="uint8")
    values["amax"][:] = rng.uniform(0.5, 3, p["experts"])
    values["denominator"][:] = 2688
    rows, depth = p["rows"], p["depth"]
    selected = np.arange(p["start"], p["start"] + p["count"])
    out = decode_rows(
        values["w"].reshape(-1, depth // 2)[selected],
        values["scale"].reshape(-1, depth // 16)[selected],
        values["amax"][selected // rows, None],
        values["denominator"],
    )
    return ValidationCase(values, {**values, "out": out}, atol=2e-7, rtol=2e-7)
