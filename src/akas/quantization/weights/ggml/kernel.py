"""Shared TileLang computations for ggml."""


def weight_decoder(kind, depth, *, packed_words=False):
    """Inline the exact GGML scalar decoder into a caller's TIRx function."""
    if kind == 30:
        return bf16_decoder(depth, packed_words=packed_words)
    if kind == 3:
        return q4_1_decoder(depth, packed_words=packed_words)
    if kind == 13:
        return q5_decoder(depth, packed_words=packed_words)
    import tilelang.language as T
    from akas.quantization.weights.ggml.definition import ENCODINGS as TYPES

    _, block, size = TYPES[kind]

    @T.macro
    def byte(w, base, offset):
        if packed_words:
            return w[(base + offset) // 4] >> (base + offset) % 4 * 8 & T.uint32(255)
        else:
            return T.cast(w[base + offset], "uint32")

    @T.macro
    def half(w, base, offset):
        bits = byte(w, base, offset) | byte(w, base, offset + 1) << 8
        if packed_words:
            return T.call_extern("float32", "tensor_unpack_f16", bits)
        else:
            return T.cast(T.reinterpret("float16", T.cast(bits, "uint16")), "float32")

    @T.macro
    def signed(value):
        if packed_words:
            return T.cast(
                T.cast(value, "int32")
                - T.if_then_else(T.cast(value, "int32") >= 128, 256, 0),
                "float32",
            )
        else:
            return T.cast(T.reinterpret("int8", T.cast(value, "uint8")), "float32")

    @T.macro
    def read_weight(w, row, col):
        if kind in (0, 1):
            return T.cast(w[row * depth + col], "float32")
        else:
            base = (row * depth + col) // block * size
            j = col % block
            if kind == 2:
                return half(w, base, 0) * (
                    T.cast(byte(w, base, 2 + j % 16) >> 4 * (j // 16) & 15, "float32")
                    - 8
                )
            elif kind == 8:
                return half(w, base, 0) * signed(byte(w, base, 2 + j))
            elif kind == 12:
                group = j // 32
                scale = T.if_then_else(
                    group < 4,
                    byte(w, base, 4 + group) & 63,
                    byte(w, base, 8 + group) & 15 | byte(w, base, group) >> 6 << 4,
                )
                minimum = T.if_then_else(
                    group < 4,
                    byte(w, base, 8 + group) & 63,
                    byte(w, base, 8 + group) >> 4 | byte(w, base, 4 + group) >> 6 << 4,
                )
                quant = (
                    byte(w, base, 16 + j // 64 * 32 + j % 32) >> 4 * (group % 2) & 15
                )
                return half(w, base, 0) * T.cast(scale, "float32") * T.cast(
                    quant, "float32"
                ) - half(w, base, 2) * T.cast(minimum, "float32")
            elif kind == 14:
                group = j % 128 // 32
                low = byte(w, base, j // 128 * 64 + group % 2 * 32 + j % 32)
                high = byte(w, base, 128 + j // 128 * 32 + j % 32)
                quant = low >> 4 * (group // 2) & 15 | (high >> 2 * group & 3) << 4
                scale = byte(w, base, 192 + j // 16)
                return (
                    half(w, base, 208) * signed(scale) * (T.cast(quant, "float32") - 32)
                )
            else:
                raise ValueError("unsupported GGML encoding")

    return read_weight


def bf16_decoder(depth, *, packed_words=False):
    """Decode a BF16 word directly into its FP32 bit representation."""
    import tilelang.language as T

    @T.macro
    def read_bf16(w, row, col):
        index = row * depth + col
        if packed_words:
            bits = (w[index // 2] >> (index % 2 * 16)) & T.uint32(65535)
        else:
            bits = T.cast(w[index * 2], "uint32") | (
                T.cast(w[index * 2 + 1], "uint32") << 8
            )
        return T.reinterpret("float32", bits << 16)

    return read_bf16


def q4_1_decoder(depth, *, packed_words=False):
    """Affine nibble blocks, additive to the retained GGML inline computations."""
    import tilelang.language as T

    @T.macro
    def byte(w, base, offset):
        if packed_words:
            return w[(base + offset) // 4] >> (base + offset) % 4 * 8 & T.uint32(255)
        else:
            return T.cast(w[base + offset], "uint32")

    @T.macro
    def half(w, base, offset):
        bits = byte(w, base, offset) | byte(w, base, offset + 1) << 8
        if packed_words:
            return T.call_extern("float32", "tensor_unpack_f16", bits)
        else:
            return T.cast(T.reinterpret("float16", T.cast(bits, "uint16")), "float32")

    @T.macro
    def read_weight_q4_1(w, row, col):
        base = (row * depth + col) // 32 * 20
        j = col % 32
        return half(w, base, 0) * T.cast(
            byte(w, base, 4 + j % 16) >> 4 * (j // 16) & 15, "float32"
        ) + half(w, base, 2)

    return read_weight_q4_1


def prepack_host(data):
    """Aligned signed-byte blocks and an exact F32 scale (36 bytes per block).

    This derived cache doubles Q4_0 storage; it does not change its values.
    Eight words retain the low-16/high-16 element order, then one scale word.
    """
    import numpy as np

    raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    if raw.size % 18:
        raise ValueError("incomplete Q4_0 cache block")
    blocks = raw.reshape(-1, 18)
    result = np.empty((len(blocks), 9), np.uint32)
    signed = (
        np.concatenate((blocks[:, 2:] & 15, blocks[:, 2:] >> 4), axis=1).astype(
            np.int16
        )
        - 8
    ).astype(np.int8)
    result[:, :8] = signed.view(np.uint32)
    result[:, 8] = (
        blocks[:, :2].copy().view("<f2").astype("<f4").reshape(-1).view(np.uint32)
    )
    return result.reshape(-1)


def decode_host(data, kind):
    import numpy as np
    from .definition import ENCODINGS as TYPES

    raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    if kind not in TYPES:
        raise ValueError(f"unsupported GGML type {kind}")
    _, block, width = TYPES[kind]
    if raw.size % width:
        raise ValueError("incomplete quantized block")
    if kind in (0, 1):
        return raw.view("<f4" if kind == 0 else "<f2").astype(np.float32)
    if kind == 30:
        return (raw.view("<u2").astype("uint32") << 16).view("float32")
    x = raw.reshape(-1, width)
    if kind == 3:
        d = x[:, :2].copy().view("<f2").astype(np.float32)
        minimum = x[:, 2:4].copy().view("<f2").astype(np.float32)
        q = np.concatenate((x[:, 4:] & 15, x[:, 4:] >> 4), axis=1).astype(np.float32)
        return (d * q + minimum).reshape(-1)
    if kind in (2, 8):
        d = x[:, :2].copy().view("<f2").astype(np.float32)
        if kind == 8:
            q = x[:, 2:].view(np.int8).astype(np.float32)
        else:
            q = (
                np.concatenate((x[:, 2:] & 15, x[:, 2:] >> 4), axis=1).astype(
                    np.float32
                )
                - 8
            )
        return (d * q).reshape(-1)
    if kind in (12, 13):
        d = x[:, :2].copy().view("<f2").astype(np.float32)
        minimum = x[:, 2:4].copy().view("<f2").astype(np.float32)
        packed = x[:, 4:16]
        scales = []
        mins = []
        for j in range(8):
            scales.append(
                packed[:, j] & 63
                if j < 4
                else (packed[:, j + 4] & 15) | ((packed[:, j - 4] >> 6) << 4)
            )
            mins.append(
                packed[:, j + 4] & 63
                if j < 4
                else (packed[:, j + 4] >> 4) | ((packed[:, j] >> 6) << 4)
            )
        q = x[:, 16 if kind == 12 else 48 :].reshape(-1, 4, 32)
        q = np.stack((q & 15, q >> 4), axis=2).reshape(-1, 8, 32).astype(np.float32)
        if kind == 13:
            high = (x[:, None, 16:48] >> np.arange(8, dtype="uint8")[None, :, None]) & 1
            q += high.astype("float32") * 16
        result = (
            d[:, None, :] * np.stack(scales, axis=1)[:, :, None] * q
            - minimum[:, None, :] * np.stack(mins, axis=1)[:, :, None]
        )
        return result.reshape(-1)
    d = x[:, 208:210].copy().view("<f2").astype(np.float32)
    result = np.empty((len(x), 256), np.float32)
    for half in range(2):
        low = x[:, half * 64 : half * 64 + 64]
        high = x[:, 128 + half * 32 : 160 + half * 32]
        scales = x[:, 192 + half * 8 : 200 + half * 8].view(np.int8).astype(np.float32)
        for group in range(4):
            nibble = low[:, (group % 2) * 32 : (group % 2 + 1) * 32]
            q = ((nibble & 15) if group < 2 else nibble >> 4) | (
                ((high >> (2 * group)) & 3) << 4
            )
            scale = np.repeat(scales[:, group * 2 : group * 2 + 2], 16, axis=1)
            result[:, half * 128 + group * 32 : half * 128 + (group + 1) * 32] = (
                d * scale * (q.astype(np.float32) - 32)
            )
    return result.reshape(-1)


def dequantize(spec, schedule, *, provider):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from .definition import FORMAT

    p = {**spec.parameters, **schedule}
    rows, depth = p["r"], p["k"]
    encoding = p["type"]
    binding = FORMAT.bind(
        (rows, depth),
        provider=provider,
        encoding=encoding,
        **(dict(layout=p["weight_layout"]) if "weight_layout" in p else {}),
    )
    values = binding.storage.values
    read = binding.read
    dtype = p.get("output_dtype", "float32")
    if dtype not in ("float16", "float32"):
        raise ValueError("dequantization output must be F16 or F32")
    from akas.quantization._shared.rounding import round_half

    rounded = round_half()

    @kernel
    def compute(w: Tensor(values.shape, values.dtype)) -> Tensor((rows, depth), dtype):
        out = output((rows, depth), dtype)
        with T.Kernel(T.ceildiv(rows * depth, 128), threads=128) as block:
            for lane in T.Parallel(128):
                index = block * 128 + lane
                if index < rows * depth:
                    value = read((w,), index // depth, index % depth)
                    out[index // depth, index % depth] = (
                        rounded(value) if dtype == "float16" else value
                    )
        return out

    return compute


def q5_decoder(depth, *, packed_words=False):
    """Q5_K reader, additive to the retained optimized GGML computations."""
    import tilelang.language as T

    @T.macro
    def byte(w, base, offset):
        if packed_words:
            return w[(base + offset) // 4] >> (base + offset) % 4 * 8 & T.uint32(255)
        else:
            return T.cast(w[base + offset], "uint32")

    @T.macro
    def half(w, base, offset):
        bits = byte(w, base, offset) | byte(w, base, offset + 1) << 8
        if packed_words:
            return T.call_extern("float32", "tensor_unpack_f16", bits)
        else:
            return T.cast(T.reinterpret("float16", T.cast(bits, "uint16")), "float32")

    @T.macro
    def read_weight_q5(w, row, col):
        base = (row * depth + col) // 256 * 176
        j = col % 256
        group = j // 32
        scale = T.if_then_else(
            group < 4,
            byte(w, base, 4 + group) & 63,
            byte(w, base, 8 + group) & 15 | byte(w, base, group) >> 6 << 4,
        )
        minimum = T.if_then_else(
            group < 4,
            byte(w, base, 8 + group) & 63,
            byte(w, base, 8 + group) >> 4 | byte(w, base, 4 + group) >> 6 << 4,
        )
        low = byte(w, base, 48 + j // 64 * 32 + j % 32)
        high = byte(w, base, 16 + j % 32)
        quant = (low >> (4 * (group % 2)) & 15) | ((high >> group & 1) << 4)
        return half(w, base, 0) * T.cast(scale, "float32") * T.cast(
            quant, "float32"
        ) - half(w, base, 2) * T.cast(minimum, "float32")

    return read_weight_q5


def repack_host(data, *, source="raw", target="planar"):
    """Reversibly separate Q4_0 payloads and scales without changing a bit.

    The returned contiguous array has the original byte count. Source storage
    is borrowed and never modified; at most one tensor-sized destination lives
    alongside it. This is a byte permutation, not numerical quantization.
    """
    import numpy as np

    if source not in ("raw", "planar") or target not in ("raw", "planar"):
        raise ValueError("Q4_0 layout must be raw or planar")
    raw = np.ascontiguousarray(data).view("uint8").reshape(-1)
    if not raw.size or raw.size % 18:
        raise ValueError("Q4_0 layout conversion requires whole 18-byte blocks")
    if source == target:
        return raw
    count = raw.size // 18
    result = np.empty_like(raw)
    if source == "raw":
        blocks = raw.reshape(count, 18)
        result[: count * 16].reshape(count, 16)[:] = blocks[:, 2:]
        result[count * 16 :].reshape(count, 2)[:] = blocks[:, :2]
    else:
        blocks = result.reshape(count, 18)
        blocks[:, 2:] = raw[: count * 16].reshape(count, 16)
        blocks[:, :2] = raw[count * 16 :].reshape(count, 2)
    return result


def planar_reader(shape):
    """One shared F32 reader over separated Q4_0 bytes and unchanged F16 scales."""
    from math import prod
    import tilelang.language as T
    from akas.quantization._shared.rounding import half_bits

    depth = shape[-1]
    blocks = prod(shape) // 32
    words_count = blocks * 18 // 4
    unpack = half_bits()

    @T.macro
    def read(encoded, row, column):
        words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
        block = row * (depth // 32) + column // 32
        scale_address = blocks * 16 + block * 2
        scale = unpack(
            words[scale_address // 4] >> (scale_address % 4 * 8) & T.uint32(65535)
        )
        packed = words[block * 4 + column % 16 // 4]
        bits = packed >> (column % 4 * 8 + column % 32 // 16 * 4) & T.uint32(15)
        return (T.cast(bits, "float32") - 8) * scale

    return read
