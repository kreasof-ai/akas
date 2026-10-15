"""Inline adapters shared by representations; no operation scheduling lives here."""


def permuted_reader(read, row_order=None, column_order=None):
    """Read grouped logical axes from a checkpoint's tiled physical head order.

    An order is (prefix, groups, repeats, width). The prefix is unchanged; the
    remainder transposes the groups/repeats axes while keeping lanes contiguous.
    """
    import tilelang.language as T

    @T.macro
    def index(value, order):
        if order is None:
            return value
        else:
            start, groups, repeats, width = order
            local = value - start
            return T.if_then_else(
                value < start,
                value,
                start
                + (local // width % repeats * groups + local // (width * repeats))
                * width
                + local % width,
            )

    @T.macro
    def load(encoded, row, column):
        return read(encoded, index(row, row_order), index(column, column_order))

    return load


def restore_order(values, axis, order):
    """Host counterpart of grouped-to-tiled addressing for checkpoint inspection."""
    import numpy as np

    if order is None:
        return values
    start, groups, repeats, width = order
    positions = np.arange(values.shape[axis])
    tail = positions[start:] - start
    positions[start:] = (
        start
        + (tail // width % repeats * groups + tail // (width * repeats)) * width
        + tail % width
    )
    return np.take(values, positions, axis=axis)


def matrix_reader(load):
    """Adapt an embedded-metadata reader without changing its arithmetic."""
    import tilelang.language as T

    @T.macro
    def read(encoded, row, column):
        return load(encoded[0], row, column)

    return read


def matrix_tile(read, shape):
    """Decode once per caller-owned tile element, with zero-filled boundaries."""
    import tilelang.language as T
    from math import prod

    rows, depth = prod(shape[:-1]), shape[-1]

    @T.macro
    def read_tile(encoded, tile, row, column):
        for i, j in T.Parallel(tile.shape[0], tile.shape[1]):
            tile[i, j] = T.if_then_else(
                row + i < rows and column + j < depth,
                read(encoded, row + i, column + j),
                0,
            )

    return read_tile


def cache_reader(load):
    """Adapt an existing inline cache load to the encoded-tuple convention."""
    import tilelang.language as T

    @T.macro
    def read(encoded, slot, head, position, channel):
        return load(encoded[0], encoded[1], slot, head, position, channel)

    return read


def cache_tile(read):
    import tilelang.language as T

    @T.macro
    def read_tile(encoded, tile, slot, head, position, channel):
        for i, j in T.Parallel(tile.shape[0], tile.shape[1]):
            tile[i, j] = read(encoded, slot, head, position + i, channel + j)

    return read_tile


def cache_arguments(read):
    """Keep an existing attention ABI while its reader uses codec tuples."""
    import tilelang.language as T

    @T.macro
    def load(values, scales, slot, head, position, channel):
        return read((values, scales), slot, head, position, channel)

    return load


def packed_arguments(read):
    import tilelang.language as T

    @T.macro
    def load(values, scales, row, column):
        return read((values, scales), row, column)

    return load


def pair_arguments(write):
    import tilelang.language as T

    @T.macro
    def store(keys, values, row, head, slot, position, kc, vc, ks, vs):
        write(keys, values, row, head, slot, position, (kc, ks), (vc, vs))

    return store


def writer_arguments(write):
    import tilelang.language as T

    @T.macro
    def store(keys, values, row, head, position, kc, vc, ks, vs, tx, enabled):
        write(keys, values, row, head, position, (kc, ks), (vc, vs), tx, enabled)

    return store
