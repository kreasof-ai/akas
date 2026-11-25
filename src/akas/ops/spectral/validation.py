"""Independent NumPy references for spectral and band-layout operations."""

from akas.validation import ValidationCase, sample_arguments


def _expect(values, name, result):
    expected = {key: value.copy() for key, value in values.items()}
    expected[name][:] = result
    return ValidationCase(values, expected, rtol=2e-4, atol=2e-5)


def stft_validation(kind, p, signature):
    import numpy as np

    values = sample_arguments(signature)
    n, size, hop = (p[key] for key in ("samples", "fft", "hop"))
    center = size // 2
    audio = values["audio"]
    padded = np.pad(
        audio, center, mode="reflect" if p.get("reflect", True) else "constant"
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    result = np.fft.rfft(windows * values["window"], axis=1)
    return _expect(values, "out", np.stack((result.real, result.imag), axis=-1))


def istft_validation(kind, p, signature):
    import numpy as np

    values = sample_arguments(signature)
    if kind == "istft_matrix":
        bins = values["spectrum"].shape[1]
        frequency = np.arange(bins, dtype="float64")
        tap = np.arange(p["fft"], dtype="float64")[:, None]
        angle = 2 * np.pi * frequency[None, :] * tap / p["fft"]
        weight = np.full(bins, 2.0)
        weight[[0, -1]] = 1.0
        values["matrix"][:] = (
            np.stack((np.cos(angle), -np.sin(angle)), axis=-1).reshape(
                p["fft"], 2 * bins
            )
            * weight.repeat(2)[None, :]
            / p["fft"]
        )
        result = np.fft.irfft(
            values["spectrum"][..., 0] + 1j * values["spectrum"][..., 1],
            n=p["fft"],
            axis=-1,
        )
        return _expect(values, "out", result)
    n, size, hop = (p[key] for key in ("samples", "fft", "hop"))
    analysis = values["window"]
    result = np.zeros(n + size, dtype="float64")
    weight = np.zeros_like(result)
    for frame, waveform in enumerate(values["frames_input"]):
        start = frame * hop
        result[start : start + size] += waveform * analysis
        weight[start : start + size] += analysis**2
    correction = result[size // 2 : size // 2 + n] / np.maximum(
        weight[size // 2 : size // 2 + n], 1e-8
    )
    return _expect(values, "out", values["original"] + correction)


def complex_correction_validation(kind, p, signature):
    import numpy as np

    values = sample_arguments(signature)
    raw = values["raw"]
    spectrum = values["spectrum"]
    mask = 2 * np.tanh(raw[..., :2])
    residual = np.tanh(raw[..., 2:])
    norm = np.maximum(np.linalg.norm(residual, axis=-1, keepdims=True), 1)
    magnitude = p["bound"] * np.linalg.norm(spectrum, axis=-1, keepdims=True)
    residual = residual / norm * magnitude
    sr, si = spectrum[..., 0], spectrum[..., 1]
    mr, mi = mask[..., 0], mask[..., 1]
    result = np.stack(
        (sr * mr - si * mi + residual[..., 0], sr * mi + si * mr + residual[..., 1]),
        axis=-1,
    )
    return _expect(values, "out", result)


def exclusive_attention_validation(kind, p, signature):
    import numpy as np

    values = sample_arguments(signature)
    attended, original = values["attended"], values["values"]
    numerator = np.sum(attended * original, axis=-1, keepdims=True)
    denominator = np.maximum(np.sum(original**2, axis=-1, keepdims=True), 1e-6)
    return _expect(values, "out", attended - numerator / denominator * original)


def token_transpose_validation(kind, p, signature):
    values = sample_arguments(signature)
    return _expect(values, "out", values["x"].transpose(1, 0, 2).copy())


def band_gather_validation(kind, p, signature):
    values = sample_arguments(signature)
    return _expect(values, "out", values["x"][p["band"]].copy())


def band_scatter_validation(kind, p, signature):
    values = sample_arguments(signature)
    expected = {key: value.copy() for key, value in values.items()}
    expected["spectrum"][:, p["start"] : p["end"]] = values["x"]
    return ValidationCase(values, expected, rtol=0, atol=0)


def spectrum_band_validation(kind, p, signature):
    values = sample_arguments(signature)
    return _expect(
        values, "out", values["x"][:, p["start"] : p["end"]].reshape(p["frames"], -1)
    )


def token_scatter_validation(kind, p, signature):
    values = sample_arguments(signature)
    expected = {key: value.copy() for key, value in values.items()}
    expected["features"][p["band"]] = values["x"] + values["embedding"]
    return ValidationCase(values, expected, rtol=0, atol=0)


def channel_slice_validation(kind, p, signature):
    values = sample_arguments(signature)
    return _expect(
        values, "out", values["x"][:, p["start"] : p["start"] + p["width"]].copy()
    )


def channel_affine_validation(kind, p, signature):
    values = sample_arguments(signature)
    result = values["x"] * (1 + values["scale"]) + values["shift"]
    return _expect(values, "out", result)


def elementwise_binary_validation(kind, p, signature):
    import numpy as np

    values = sample_arguments(signature)
    if p["operation"] == "multiply":
        result = values["x"] * values["y"]
    elif p["operation"] == "divide":
        result = values["x"] / np.maximum(values["y"], 1e-6)
    else:
        result = values["x"] - values["y"]
    return _expect(values, "out", result)


VALIDATIONS = {
    "stft": stft_validation,
    "istft_matrix": istft_validation,
    "istft_overlap": istft_validation,
    "complex_correction": complex_correction_validation,
    "exclusive_attention": exclusive_attention_validation,
    "token_transpose": token_transpose_validation,
    "band_gather": band_gather_validation,
    "band_scatter": band_scatter_validation,
    "spectrum_band": spectrum_band_validation,
    "token_scatter": token_scatter_validation,
    "channel_slice": channel_slice_validation,
    "channel_affine": channel_affine_validation,
    "elementwise_binary": elementwise_binary_validation,
}
