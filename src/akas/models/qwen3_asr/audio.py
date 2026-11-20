"""PCM16 WAV decoding and Whisper log-Mel features for Qwen3 ASR."""

from functools import lru_cache
from io import BytesIO
from math import gcd
import wave

import numpy as np

SAMPLE_RATE = 16000
FFT_SIZE = 400
HOP = 160
CONV_FRAMES = 100
CONV_ROWS = (CONV_FRAMES + 7) // 8
WINDOW_CHUNKS = 8


def wav_samples(data):
    try:
        with wave.open(BytesIO(data), "rb") as source:
            channels = source.getnchannels()
            rate = source.getframerate()
            frames = source.getnframes()
            if source.getsampwidth() != 2 or not frames:
                raise ValueError("requires nonempty PCM16 WAV")
            pcm = source.readframes(frames)
            if len(pcm) != frames * channels * 2:
                raise ValueError("truncated PCM16 WAV")
    except (wave.Error, EOFError) as error:
        raise ValueError("requires nonempty PCM16 WAV") from error
    values = np.frombuffer(pcm, dtype="<i2")
    samples = (
        values.astype("float32")
        if channels == 1
        else values.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    )
    return samples * np.float32(1 / 32768), rate


@lru_cache(maxsize=8)
def _resample_phases(rate):
    radius = 24
    divisor = gcd(rate, SAMPLE_RATE)
    phases, advance = SAMPLE_RATE // divisor, rate // divisor
    centers, fraction = np.divmod(np.arange(phases) * advance, phases)
    distance = fraction[:, None] / phases - np.arange(-radius + 1, radius + 1)
    cutoff = min(1, SAMPLE_RATE / rate)
    taps = cutoff * np.sinc(cutoff * distance)
    taps *= 0.5 + 0.5 * np.cos(np.pi * distance / radius)
    return radius, advance, centers, taps


def resample(samples, rate):
    """Apply the windowed-sinc filter once per rational sample-rate phase."""
    if rate == SAMPLE_RATE:
        return samples
    radius, advance, centers, taps = _resample_phases(rate)
    phases = len(centers)
    count = (len(samples) * SAMPLE_RATE + rate - 1) // rate
    padded = np.pad(samples, (radius - 1, radius))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius)
    result = np.empty(count, dtype="float32")
    if phases <= 256:
        for phase, first in enumerate(centers[:count]):
            destination = result[phase::phases]
            source = windows[first : first + len(destination) * advance : advance]
            np.einsum("ij,j->i", source, taps[phase], out=destination, casting="unsafe")
        return result
    cycles, tail = divmod(count, phases)
    for start in range(0, cycles, max(1, SAMPLE_RATE // phases)):
        stop = min(start + max(1, SAMPLE_RATE // phases), cycles)
        indices = np.arange(start, stop)[:, None] * advance + centers
        result[start * phases : stop * phases] = np.einsum(
            "cpk,pk->cp", windows[indices], taps
        ).reshape(-1)
    if tail:
        result[cycles * phases :] = np.einsum(
            "pk,pk->p", windows[cycles * advance + centers[:tail]], taps[:tail]
        )
    return result


@lru_cache(maxsize=1)
def _filters(mels):
    linear = 200 / 3
    transition = 1000 / linear
    step = np.log(6.4) / 27
    mel_end = transition + np.log((SAMPLE_RATE / 2) / 1000) / step
    points = np.linspace(0, mel_end, mels + 2)
    hz = np.where(
        points < transition,
        points * linear,
        1000 * np.exp(step * (points - transition)),
    )
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    left = (freqs[:, None] - hz[:-2]) / (hz[1:-1] - hz[:-2])
    right = (hz[2:] - freqs[:, None]) / (hz[2:] - hz[1:-1])
    dense = (np.maximum(0, np.minimum(left, right)) * (2 / (hz[2:] - hz[:-2]))).T
    spans = [np.flatnonzero(band) for band in dense]
    bins = np.zeros((mels, max(map(len, spans))), dtype=np.intp)
    coefficients = np.zeros(bins.shape, dtype="float32")
    for band, indices in enumerate(spans):
        bins[band, : len(indices)] = indices
        coefficients[band, : len(indices)] = dense[band, indices]
    return bins, coefficients


_HANN = np.hanning(FFT_SIZE + 1)[:-1]


def log_mel(samples, mels):
    # Match the centered periodic Hann STFT used by the qwen3a projector.
    pad = FFT_SIZE // 2
    if len(samples) <= pad:
        samples = np.pad(samples, (0, pad + 1 - len(samples)))
    centered = np.pad(samples, pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(centered, FFT_SIZE)[::HOP]
    spectrum = np.fft.rfft(windows * _HANN, axis=1)
    energy = (spectrum.real**2 + spectrum.imag**2).astype("float32")
    bins, coefficients = _filters(mels)
    mel = np.einsum("fmk,mk->fm", energy[:, bins], coefficients)
    log = np.log10(np.maximum(mel, np.float32(1e-10)))
    return (np.maximum(log, log.max() - np.float32(8)) + np.float32(4)) / np.float32(4)
