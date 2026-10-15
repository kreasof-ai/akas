"""Shared TileLang computations for linear."""

from akas.kernels.kernel import Tensor, kernel, output
from collections.abc import Mapping
from akas.kernels.definition import KernelSpec


def projection(spec, schedule, *, provider):
    """Tiled fused projection over independently encoded weight pieces."""
    import tilelang.language as T
    from akas.quantization import Codec, Storage
    from akas.quantization._shared.access import matrix_tile
    from akas.quantization.weights import bind_parts
    from akas.memory.state import BufferSpec

    p = {**spec.parameters, **schedule}
    rows, depth, columns = p["r"], p["k"], p["o"]
    dtype = p.get("input_dtype", "bfloat16")
    right = bind_parts(p["weight_parts"], depth, provider=provider)

    @T.macro
    def read_left(encoded, row, column):
        return T.cast(encoded[0][row, column], "float32")

    left = Codec(
        Storage(BufferSpec((rows, depth), dtype)),
        read_left,
        matrix_tile(read_left, (rows, depth)),
    )
    algorithm = _quantized_matmul(p, left, right, provider=provider)
    buffers = right.storage.buffers
    count = len(buffers)
    if count == 1:
        w0 = buffers[0]

        @kernel
        def compute(
            x: Tensor((rows, depth), dtype), w: Tensor(w0.shape, w0.dtype)
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x,), (w,), out)
            return out

    elif count == 2:
        w0, w1 = buffers

        @kernel
        def compute(
            x: Tensor((rows, depth), dtype),
            w: Tensor(w0.shape, w0.dtype),
            weight1: Tensor(w1.shape, w1.dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x,), (w, weight1), out)
            return out

    elif count == 3:
        w0, w1, w2 = buffers

        @kernel
        def compute(
            x: Tensor((rows, depth), dtype),
            w: Tensor(w0.shape, w0.dtype),
            weight1: Tensor(w1.shape, w1.dtype),
            weight2: Tensor(w2.shape, w2.dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x,), (w, weight1, weight2), out)
            return out

    else:
        w0, w1, w2, w3 = buffers

        @kernel
        def compute(
            x: Tensor((rows, depth), dtype),
            w: Tensor(w0.shape, w0.dtype),
            weight1: Tensor(w1.shape, w1.dtype),
            weight2: Tensor(w2.shape, w2.dtype),
            weight3: Tensor(w3.shape, w3.dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x,), (w, weight1, weight2, weight3), out)
            return out

    return compute


def _quantized_matmul(p, left, right, *, provider):
    """Stage each decoded tile once; the provider owns its compute schedule."""
    import tilelang.language as T

    rows, depth, columns = p["r"], p["k"], p["o"]
    if rows == 1:
        read_left, read_right = left.read, right.read

        @T.macro
        def gemv(x, w, out):
            with T.Kernel(T.ceildiv(columns, 4), threads=128) as block:
                tx = T.get_thread_binding()
                row, lane = tx // 32, tx % 32
                source = T.alloc_shared((128,), "float16")
                scratch = T.alloc_shared((128,), "float32")
                accum = T.alloc_var("float32")
                accum = 0
                for tile in T.serial(T.ceildiv(depth, 128)):
                    source[tx] = T.if_then_else(
                        tile * 128 + tx < depth, read_left(x, 0, tile * 128 + tx), 0
                    )
                    T.sync_threads()
                    if block * 4 + row < columns:
                        for step in T.unroll(4):
                            channel = tile * 128 + step * 32 + lane
                            if channel < depth:
                                value = T.cast(
                                    read_right(w, block * 4 + row, channel), "float16"
                                )
                                accum = accum + T.cast(
                                    source[step * 32 + lane], "float32"
                                ) * T.cast(value, "float32")
                    T.sync_threads()
                scratch[tx] = accum
                T.sync_threads()
                for step in T.unroll(5):
                    if lane < (16 >> step):
                        scratch[tx] = scratch[tx] + scratch[tx + (16 >> step)]
                    T.sync_threads()
                if lane == 0 and block * 4 + row < columns:
                    out[0, block * 4 + row] = scratch[tx]

        return gemv
    defaults = (32, 64, 32) if provider == "cuda" else (16, 32, 32)
    bm, bn, bk = (
        p.get(name, value)
        for name, value in zip(("block_m", "block_n", "block_k"), defaults)
    )
    load_left, load_right = left.require("read_tile"), right.require("read_tile")

    @T.macro
    def matmul(x, w, out):
        with T.Kernel(T.ceildiv(rows, bm), T.ceildiv(columns, bn), threads=128) as (
            by,
            bx,
        ):
            lhs = T.alloc_shared((bm, bk), "float16")
            rhs = T.alloc_shared((bn, bk), "float16")
            if provider == "cuda":
                accum = T.alloc_fragment((bm, bn), "float32")
                T.clear(accum)
                for block in T.Pipelined(T.ceildiv(depth, bk), num_stages=2):
                    load_left(x, lhs, by * bm, block * bk)
                    load_right(w, rhs, bx * bn, block * bk)
                    T.gemm(lhs, rhs, accum, transpose_B=True)
                for i, j in T.Parallel(bm, bn):
                    if by * bm + i < rows and bx * bn + j < columns:
                        out[by * bm + i, bx * bn + j] = accum[i, j]
            else:
                tx = T.get_thread_binding()
                row, column = tx // 16 * 2, tx % 16 * 2
                accum = T.alloc_local((2, 2), "float32")
                T.clear(accum)
                for block in T.serial(T.ceildiv(depth, bk)):
                    load_left(x, lhs, by * bm, block * bk)
                    load_right(w, rhs, bx * bn, block * bk)
                    T.sync_threads()
                    for k in T.serial(bk):
                        for i in T.unroll(2):
                            for j in T.unroll(2):
                                accum[i, j] = accum[i, j] + T.cast(
                                    lhs[row + i, k], "float32"
                                ) * T.cast(rhs[column + j, k], "float32")
                    T.sync_threads()
                for i in T.unroll(2):
                    for j in T.unroll(2):
                        if by * bm + row + i < rows and bx * bn + column + j < columns:
                            out[by * bm + row + i, bx * bn + column + j] = accum[i, j]

    return matmul


def quantized_linear(spec: KernelSpec, schedule: Mapping[str, object], *, provider):
    """Explicit encoded operands; compatible codecs fuse at tile loads.

    This additive operation leaves all existing tuned GGML/FP8 choices intact.
    Values decode to F32, stage as FP16, and accumulate in F32 on both providers.
    """
    import tilelang.language as T
    from akas.memory.state import BufferSpec
    from akas.quantization import Codec, Storage, get
    from akas.quantization._shared.access import matrix_tile
    from akas.quantization.weights import bind

    p = {**spec.parameters, **schedule}
    rows, depth, columns = p["r"], p["k"], p["o"]
    activation = p.get("activation_format")
    if activation is None:

        @T.macro
        def read_float(encoded, row, column):
            return encoded[0][row, column]

        left = Codec(
            Storage(BufferSpec((rows, depth), "float32")),
            read_float,
            matrix_tile(read_float, (rows, depth)),
        )
    else:
        left = get("activations", activation).bind((rows, depth), provider=provider)
    right = bind(p, (columns, depth), provider=provider)
    x_spec, w_spec = left.storage.values, right.storage.values
    xm, wm = left.read_buffers[1:], right.read_buffers[1:]
    if len(xm) > 2 or len(wm) > 1:
        raise ValueError(
            "quantized_linear accepts up to two activation and one weight metadata buffers"
        )
    algorithm = _quantized_matmul(p, left, right, provider=provider)
    if len(xm) == 2 and wm:

        @kernel
        def compute(
            x: Tensor(x_spec.shape, x_spec.dtype),
            activation_scales: Tensor(xm[0].shape, xm[0].dtype),
            sums: Tensor(xm[1].shape, xm[1].dtype),
            w: Tensor(w_spec.shape, w_spec.dtype),
            weight_scales: Tensor(wm[0].shape, wm[0].dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x, activation_scales, sums), (w, weight_scales), out)
            return out

        return compute
    if len(xm) == 2:

        @kernel
        def compute(
            x: Tensor(x_spec.shape, x_spec.dtype),
            activation_scales: Tensor(xm[0].shape, xm[0].dtype),
            sums: Tensor(xm[1].shape, xm[1].dtype),
            w: Tensor(w_spec.shape, w_spec.dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x, activation_scales, sums), (w,), out)
            return out

        return compute
    if xm and wm:

        @kernel
        def compute(
            x: Tensor(x_spec.shape, x_spec.dtype),
            activation_scales: Tensor(xm[0].shape, xm[0].dtype),
            w: Tensor(w_spec.shape, w_spec.dtype),
            weight_scales: Tensor(wm[0].shape, wm[0].dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x, activation_scales), (w, weight_scales), out)
            return out

        return compute
    if xm:

        @kernel
        def compute(
            x: Tensor(x_spec.shape, x_spec.dtype),
            activation_scales: Tensor(xm[0].shape, xm[0].dtype),
            w: Tensor(w_spec.shape, w_spec.dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x, activation_scales), (w,), out)
            return out

        return compute
    if wm:

        @kernel
        def compute(
            x: Tensor(x_spec.shape, x_spec.dtype),
            w: Tensor(w_spec.shape, w_spec.dtype),
            weight_scales: Tensor(wm[0].shape, wm[0].dtype),
        ) -> Tensor((rows, columns), "float32"):
            out = output((rows, columns), "float32")
            algorithm((x,), (w, weight_scales), out)
            return out

        return compute

    @kernel
    def compute(
        x: Tensor(x_spec.shape, x_spec.dtype), w: Tensor(w_spec.shape, w_spec.dtype)
    ) -> Tensor((rows, columns), "float32"):
        out = output((rows, columns), "float32")
        algorithm((x,), (w,), out)
        return out

    return compute


def cuda_abi(p, algorithm, paired=False, add_residual=False):
    """Select the fused operator ABI without generating Python signatures."""
    import tilelang.language as T
    from akas.quantization.weights.ggml.definition import weight_storage

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    count, dtype = weight_storage(q, k * o)
    if paired:

        @kernel
        def compute(
            x: Tensor(r * k, "float32"),
            w: Tensor(count, dtype),
            w2: Tensor(count, dtype),
        ) -> Tensor(r * o, "float32"):
            out = output(r * o, "float32")
            algorithm(x, w, out, w2)
            return out

        return compute
    if add_residual:

        @kernel
        def compute(
            x: Tensor(r * k, "float32"),
            w: Tensor(count, dtype),
            residual: Tensor(r * o, "float32"),
        ) -> Tensor(r * o, "float32"):
            out = output(r * o, "float32")
            algorithm(x, w, out, None, residual)
            return out

        return compute

    @kernel
    def compute(x: Tensor(r * k, "float32"), w: Tensor(count, dtype)) -> Tensor(
        r * o, "float32"
    ):
        out = output(r * o, "float32")
        algorithm(x, w, out)
        return out

    return compute


def webgpu_abi(p, algorithm, paired=False, add_residual=False):
    import tilelang.language as T
    from akas.quantization.weights.ggml.definition import weight_storage

    r, k, o, q = (p.get("r", 1), p["k"], p["o"], p["type"])
    count, dtype = weight_storage(q, k * o, packed_words=True)
    if paired:

        @kernel
        def compute(
            x: Tensor(r * k, "float32"),
            w: Tensor(count, dtype),
            w2: Tensor(count, dtype),
        ) -> Tensor(r * o, "float32"):
            out = output(r * o, "float32")
            algorithm(x, w, out, w2)
            return out

        return compute
    if add_residual:

        @kernel
        def compute(
            x: Tensor(r * k, "float32"),
            w: Tensor(count, dtype),
            residual: Tensor(r * o, "float32"),
        ) -> Tensor(r * o, "float32"):
            out = output(r * o, "float32")
            algorithm(x, w, out, None, residual)
            return out

        return compute

    @kernel
    def compute(x: Tensor(r * k, "float32"), w: Tensor(count, dtype)) -> Tensor(
        r * o, "float32"
    ):
        out = output(r * o, "float32")
        algorithm(x, w, out)
        return out

    return compute


def _epilogue(paired=False, add_residual=False):
    import tilelang.language as T

    @T.macro
    def store(out, residual, index, gate, up):
        if paired:
            out[index] = gate / (1 + T.exp(-gate)) * up
        elif add_residual:
            out[index] = residual[index] + gate
        else:
            out[index] = gate

    return store


def linear(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r, k, o = (p["r"], p["k"], p["o"])
    threads = p.get("threads", 64)
    if threads not in (32, 64, 128, 256):
        raise ValueError("reference projection threads must be 32/64/128/256")
    stride, offset = (p.get("output_stride", o), p.get("output_offset", 0))
    out_shape = (r, stride) if p.get("matrix_output", False) else (r * stride,)
    reduce = tree_reduce(threads)

    @kernel
    def compute(
        x: Tensor((r * k,), "float32"),
        w: Tensor((o * k,), p.get("weight_dtype", "float32")),
    ) -> Tensor(out_shape, "float32"):
        out = output(out_shape, "float32")
        with T.Kernel(T.ceildiv(o, 4), r, threads=threads) as (block, row):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((threads,), "float32")
            acc = T.alloc_var("float32")
            for group in T.serial(4):
                col = block * 4 + group
                if col < o:
                    acc = 0
                    for tile in T.serial(T.ceildiv(k, threads)):
                        if tile * threads + tx < k:
                            acc += (
                                x[row * k + tile * threads + tx]
                                * w[col * k + tile * threads + tx]
                            )
                    scratch[tx] = acc
                    T.sync_threads()
                    reduce(scratch, tx)
                    if tx == 0:
                        if p.get("matrix_output", False):
                            out[row, offset + col] = scratch[0]
                        else:
                            out[row * stride + offset + col] = scratch[0]
                    T.sync_threads()
        return out

    return compute


def packed_linear(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.weights import bind

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    k, o, q = (p["k"], p["o"], p["type"])
    weights = bind(p, (o, k), provider="cuda")
    if weights.storage.metadata:
        raise ValueError("this packed operation ABI requires embedded metadata")
    count, dtype = (weights.storage.values.shape[0], weights.storage.values.dtype)
    read_weight = weights.read
    if r == 1:

        @kernel
        def compute(
            x: Tensor((r * k,), "float32"), w: Tensor((count,), dtype)
        ) -> Tensor((r * o,), "float32"):
            out = output((r * o,), "float32")
            with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
                accum = T.alloc_fragment((4, 32), "float32")
                total = T.alloc_fragment((4,), "float32")
                T.clear(accum)
                for tile in T.serial(k // 32):
                    for row, lane in T.Parallel(4, 32):
                        accum[row, lane] += x[tile * 32 + lane] * read_weight(
                            (w,), bx * 4 + row, tile * 32 + lane
                        )
                T.reduce_sum(accum, total, dim=1)
                for row in T.Parallel(4):
                    if bx * 4 + row < o:
                        out[bx * 4 + row] = total[row]
            return out

        return compute

    @kernel
    def compute(x: Tensor((r * k,), "float32"), w: Tensor((count,), dtype)) -> Tensor(
        (r * o,), "float32"
    ):
        out = output((r * o,), "float32")
        with T.Kernel(T.ceildiv(r, 32), T.ceildiv(o, 64), threads=128) as (by, bx):
            lhs = T.alloc_shared((32, 32), "float16")
            rhs = T.alloc_shared((64, 32), "float16")
            accum = T.alloc_fragment((32, 64), "float32")
            T.clear(accum)
            for tile in T.Pipelined(k // 32, num_stages=2):
                for i, j in T.Parallel(32, 32):
                    lhs[i, j] = T.if_then_else(
                        by * 32 + i < r, x[(by * 32 + i) * k + tile * 32 + j], 0
                    )
                for i, j in T.Parallel(64, 32):
                    if bx * 64 + i < o:
                        rhs[i, j] = read_weight((w,), bx * 64 + i, tile * 32 + j)
                    else:
                        rhs[i, j] = 0
                T.gemm(lhs, rhs, accum, transpose_B=True)
            for i, j in T.Parallel(32, 64):
                if (by * 32 + i < r) & (bx * 64 + j < o):
                    out[(by * 32 + i) * o + bx * 64 + j] = accum[i, j]
        return out

    return compute
