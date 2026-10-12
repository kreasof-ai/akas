"""Advance only live request positions after a full forward step."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p["r"]
    chunk = p.get("chunk", 1)
    if kind == "masked_copy":
        c = p["c"]

        @T.prim_func
        def masked(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((r * c,), "float32"),
            enabled: T.Tensor((r,), "int32"),
        ):
            with T.Kernel(T.ceildiv(c, 128), r, threads=128) as (block, row):
                col = block * 128 + T.get_thread_binding()
                if col < c and enabled[row] != 0:
                    out[row * c + col] = x[row * c + col]

        return masked
    if kind == "chunk_controls":

        @T.prim_func
        def controls(
            positions: T.Tensor((r,), "int32"),
            lengths: T.Tensor((r,), "int32"),
            row_positions: T.Tensor((r * chunk,), "int32"),
            enabled: T.Tensor((r * chunk,), "int32"),
        ):
            with T.Kernel(T.ceildiv(r * chunk, 64), threads=64) as block:
                tx = T.get_thread_binding()
                row = block * 64 + tx
                if row < r * chunk:
                    # Padding must never index past the cache allocation.
                    row_positions[row] = positions[row // chunk] + T.min(
                        row % chunk, T.max(lengths[row // chunk] - 1, 0)
                    )
                    enabled[row] = T.if_then_else(
                        row % chunk < lengths[row // chunk], 1, 0
                    )

        return controls
    if kind == "chunk_last":
        c = p["c"]

        @T.prim_func
        def last(
            x: T.Tensor((r * chunk * c,), "float32"),
            out: T.Tensor((r * c,), "float32"),
            lengths: T.Tensor((r,), "int32"),
        ):
            with T.Kernel(T.ceildiv(c, 64), r, threads=64) as (block, slot):
                col = block * 64 + T.get_thread_binding()
                if col < c:
                    out[slot * c + col] = x[
                        (slot * chunk + T.max(lengths[slot] - 1, 0)) * c + col
                    ]

        return last
    if kind == "next_tokens":

        @T.prim_func
        def copy(sampled: T.Tensor((r,), "int32"), tokens: T.Tensor((r,), "int32")):
            with T.Kernel(T.ceildiv(r, 64), threads=64) as block:
                tx = T.get_thread_binding()
                row = block * 64 + tx
                if row < r:
                    tokens[row] = sampled[row]

        return copy

    @T.prim_func
    def kernel(positions: T.Tensor((r,), "int32"), enabled: T.Tensor((r,), "int32")):
        with T.Kernel(T.ceildiv(r, 64), threads=64) as block:
            tx = T.get_thread_binding()
            row = block * 64 + tx
            if row < r:
                positions[row] += (
                    enabled[row]
                    if chunk > 1
                    else T.if_then_else(enabled[row] != 0, 1, 0)
                )

    return kernel
