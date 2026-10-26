"""Shared FP8 weights access and standalone dequantization."""


def reader(shape, *, provider, scale_layout="block"):
    from akas.quantization._shared.fp8 import reader

    if scale_layout == "channel":
        return channel_reader(shape, provider=provider)
    return reader(shape, role="weights", provider=provider)


def channel_reader(shape, *, provider):
    """Read E4M3 bytes with one BF16 scale per output channel."""
    import tilelang.language as T

    if provider != "cuda" or len(shape) != 2:
        raise ValueError("channel-scaled FP8 currently requires a CUDA matrix")

    @T.macro
    def read(encoded, row, column):
        return T.call_extern(
            "float32", "tensor_decode_e4m3", T.cast(encoded[0][row, column], "uint32")
        ) * T.cast(encoded[1][row, 0], "float32")

    return read


def dequantize(spec, schedule, *, provider):
    from akas.quantization._shared.fp8 import dequantize

    return dequantize(spec, schedule, role="weights", provider=provider)


def prepare_host(bits, scales, *, provider):
    """Pack finite E4M3 bytes and preserve the representation's BF16 scales.

    ``bits`` has its logical matrix/expert shape; ``scales`` contains numerical
    floating-point values, not raw BF16 bytes. Returned arrays follow the named
    storage buffers and can be uploaded using each BufferSpec's dtype.
    """
    import numpy as np
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
    from akas.quantization._shared.packing import pack_words
    from .definition import FORMAT

    bits = np.ascontiguousarray(bits, dtype="uint8")
    storage = FORMAT.storage(bits.shape, provider=provider)
    if np.any((bits & 127) == 127):
        raise ValueError("resident FP8 storage must be finite")
    scales = np.asarray(scales, dtype="float32")
    if scales.shape != storage.named["scales"].shape:
        raise ValueError("FP8 scales differ from the declared block layout")
    scales = decode_bfloat16(encode_bfloat16(scales))
    values = (
        pack_words(bits).reshape(storage.values.shape)
        if provider == "webgpu"
        else bits.copy()
    )
    return dict(values=values, scales=scales)
