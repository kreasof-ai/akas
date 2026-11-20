"""PCM WAV input and unnormalized FP32 WAV output for voice enhancement."""

from io import BytesIO
import struct
import wave

import numpy as np


SAMPLE_RATE = 48000


def wav_samples(data):
    floating = False
    try:
        with wave.open(BytesIO(data), "rb") as source:
            channels, rate = source.getnchannels(), source.getframerate()
            width, frames = source.getsampwidth(), source.getnframes()
            pcm = source.readframes(frames)
    except (wave.Error, EOFError) as error:
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ValueError("requires a PCM WAV recording") from error
        offset = 12
        channels = width = rate = frames = None
        floating = False
        while offset + 8 <= len(data):
            name, length = (
                data[offset : offset + 4],
                struct.unpack_from("<I", data, offset + 4)[0],
            )
            payload = offset + 8
            if payload + length > len(data):
                raise ValueError("truncated WAV chunk") from error
            if name == b"fmt " and length >= 16:
                codec, channels, rate, _, _, width, _ = struct.unpack_from(
                    "<HHIIHHH", data, payload
                )
                floating = codec == 3 and width == 32
                if codec != 3 and codec not in (1, 0xFFFE):
                    raise ValueError("requires PCM WAV") from error
                width //= 8
            elif name == b"data":
                pcm = data[payload : payload + length]
                frames = len(pcm) // (width * channels)
                break
            offset = payload + length + (length % 2)
        if not frames:
            raise ValueError(
                "requires nonempty 48 kHz PCM16, PCM32 or FP32 WAV"
            ) from error
    if (
        width not in (2, 4)
        or not frames
        or rate != SAMPLE_RATE
        or (floating and width != 4)
    ):
        raise ValueError("requires nonempty 48 kHz PCM16, PCM32 or FP32 WAV")
    values = np.frombuffer(
        pcm, dtype="<f4" if floating else "<i2" if width == 2 else "<i4"
    )
    samples = (
        values.astype("float32")
        if floating
        else values.astype("float32") / (32768 if width == 2 else 2147483648)
    )
    if channels != 1:
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("audio contains NaN or infinite samples")
    return samples, rate


def float_wav(samples, sample_rate=SAMPLE_RATE):
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Voice Isolator output is 48 kHz")
    values = np.ascontiguousarray(samples, dtype="<f4")
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("enhanced audio must be finite and nonempty")
    header = BytesIO()
    data_bytes = values.nbytes
    header.write(b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE")
    header.write(
        b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, sample_rate, sample_rate * 4, 4, 32)
    )
    header.write(b"data" + struct.pack("<I", data_bytes))
    header.write(values.tobytes())
    return header.getvalue()
