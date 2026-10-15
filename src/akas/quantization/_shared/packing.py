"""Storage codec implementation and independent arithmetic."""


def pack_words(data):
    """Preserve bytes and zero-pad the final physical WebGPU storage word."""
    import numpy as np

    raw = np.ascontiguousarray(data, dtype="uint8").reshape(-1)
    if raw.size % 4:
        raw = np.pad(raw, (0, -raw.size % 4))
    return raw.view("uint32")


def packed_loads():
    import tilelang.language as T

    from akas.quantization._shared.rounding import half_bits

    unpack = half_bits()

    @T.macro
    def byte(w, address):
        return w[address // 4] >> address % 4 * 8 & T.uint32(255)

    @T.macro
    def half(w, address):
        return unpack(w[address // 4] >> address % 4 * 8 & T.uint32(65535))

    @T.macro
    def word(w, address):
        # The 18-byte GGML Q4 blocks alternate between 2- and 4-byte alignment.
        return T.call_extern(
            "uint32",
            "tensor_byte_align_u32",
            w[address // 4],
            w[(address + 3) // 4],
            T.cast(address % 4, "uint32"),
        )

    return (byte, half, word)
