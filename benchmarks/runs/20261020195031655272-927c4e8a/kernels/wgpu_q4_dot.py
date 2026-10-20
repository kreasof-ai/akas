"""Quantization prototype: original Q4 bytes, packed F32 lane products."""
from dataclasses import replace


def bind(parts, depth):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import vector
    from akas.quantization.weights import bind_parts
    from akas.quantization._shared.packing import packed_loads

    if (len(parts) != 1 or parts[0]['format'] != 'ggml'
            or parts[0]['options'] != {'encoding': 2}
            or parts[0].get('row_order') or parts[0].get('column_order') or depth % 256):
        raise ValueError('Research packed lane product requires one aligned Q4_0 piece')
    codec = bind_parts(parts, depth, provider='webgpu')
    byte, half, word = packed_loads()
    row_bytes = depth // 32 * 18

    def activation(x, channel):
        return vector([x[channel + i] for i in range(4)])

    def coefficients(packed, scale, offset):
        return vector([(T.cast((packed >> (i * 8 + offset)) & 15, 'float32') - 8) * scale for i in range(4)])

    @T.macro
    def dot(encoded, x, offset, tile, lane, row):
        base = row * row_bytes + (tile * 8 + lane // 4) * 18
        scale = half(encoded[0], base)
        packed = word(encoded[0], base + 2 + lane % 4 * 4)
        channel = offset + tile * 256 + lane // 4 * 32 + lane % 4 * 4
        lo = activation(x, channel)
        hi = activation(x, channel + 16)
        low = coefficients(packed, scale, 0)
        high = coefficients(packed, scale, 4)
        return T.call_extern('float32', 'dot', lo, low) + T.call_extern('float32', 'dot', hi, high)

    return replace(codec, dot=dot)
