"""CUDA staging for packed weight operands with separate block scales."""


def aligned_pair_tile(
    shape, values_per_word, group, decode_scale, decode_pair, *, lookup=None
):
    """Stage packed words and scales, then read pairs in MMA lane order."""
    import tilelang.language as T
    from akas.memory.state import BufferSpec
    from akas.quantization.formats import StagedTile

    experts, rows, depth = shape
    bits = 32 // values_per_word

    def tile(tile_shape, dtype):
        height, width = tile_shape
        if dtype != "bfloat16" or rows % height or depth % width or width % group:
            raise ValueError("packed MMA needs whole tiles and scale blocks")
        packed_width = width // values_per_word
        scale_width = width // group
        storage = BufferSpec((height, packed_width), "uint32")
        metadata = BufferSpec((height, scale_width), "uint8" if lookup else "float32")

        @T.macro
        def load(encoded, packed, scales, row, channel):
            words = T.meta_var(
                T.Tensor(
                    (experts * rows, depth // values_per_word),
                    "uint32",
                    encoded[0].data,
                )
            )
            blocks = T.meta_var(
                T.Tensor((experts * rows, depth // group), "uint8", encoded[1].data)
            )
            start = channel // values_per_word
            group_start = channel // group
            T.copy(words[row : row + height, start : start + packed_width], packed)
            for i, j in T.Parallel(height, scale_width):
                scales[i, j] = decode_scale(blocks[row + i, group_start + j])

        @T.macro
        def pair(packed, scales, cache, row, channel):
            word = packed[row, channel // values_per_word]
            scale = scales[row, channel // group]
            return decode_pair(
                word >> ((channel % values_per_word) * bits), scale, cache
            )

        return StagedTile(storage, load, pair, metadata=metadata, lookup=lookup)

    return tile
