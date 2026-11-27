"""Independent host reconstruction of MXFP8 weights."""


def decode_rows(values, scales):
    import numpy as np
    from akas.quantization.host import decode_storage

    exponents = scales.astype("int32") - 127
    return np.ldexp(decode_storage(values), np.repeat(exponents, 32, axis=-1))


def validation(kind, p, signature):
    import numpy as np
    from akas.validation import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    rng = np.random.default_rng(73)
    values["w"][:] = rng.integers(0, 254, values["w"].shape, dtype="uint8")
    values["w"][values["w"] == 127] = 126
    values["scale"][:] = rng.integers(116, 125, values["scale"].shape, dtype="uint8")
    select = slice(p["start"], p["start"] + p["count"])
    out = decode_rows(
        values["w"].reshape(-1, p["depth"])[select],
        values["scale"].reshape(-1, p["depth"] // 32)[select],
    )
    return ValidationCase(values, {**values, "out": out}, atol=0, rtol=0)
