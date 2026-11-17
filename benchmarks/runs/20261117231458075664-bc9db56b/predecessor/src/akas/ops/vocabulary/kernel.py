"""Shared TileLang computations for vocabulary."""


def quantized_embedding(spec, schedule, *, provider):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, width = p["r"], p["c"]
    dtype = p.get("output_dtype", "bfloat16")
    binding = bind_parts(p["weight_parts"], width, provider=provider)
    storage, read = binding.storage.values, binding.read

    @kernel
    def compute(
        ids: Tensor((rows,), "int32"), w: Tensor(storage.shape, storage.dtype)
    ) -> Tensor((rows, width), dtype):
        out = output((rows, width), dtype)
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    out[row, column] = T.if_then_else(
                        ids[row] >= 0 and ids[row] < p["vocab"],
                        read((w,), T.max(0, T.min(ids[row], p["vocab"] - 1)), column),
                        0,
                    )
        return out

    return compute


from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def native_argmax(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, vocab = (p["r"], p["vocab"])

    @kernel
    def compute(
        logits: Tensor((r, vocab), "float32"),
        positions: Tensor((r,), "int32"),
        active: Tensor((r,), "int32"),
    ) -> tuple[Tensor((r,), "int32"), Tensor((r,), "int32")]:
        tokens = output((r,), "int32", position=1)
        with T.Kernel(r, threads=256) as slot:
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            values = T.alloc_shared((256,), "float32")
            indices = T.alloc_shared((256,), "int32")
            best = -T.infinity("float32")
            index = vocab
            for tile in T.serial(T.ceildiv(vocab, 256)):
                i = tile * 256 + tx
                if i < vocab:
                    if (logits[slot, i] > best) | (logits[slot, i] == best) & (
                        i < index
                    ):
                        best = logits[slot, i]
                        index = i
            values[tx] = best
            indices[tx] = index
            T.sync_threads()
            for step in T.unroll(8):
                stride = 128 >> step
                if tx < stride:
                    if (values[tx + stride] > values[tx]) | (
                        values[tx + stride] == values[tx]
                    ) & (indices[tx + stride] < indices[tx]):
                        values[tx] = values[tx + stride]
                        indices[tx] = indices[tx + stride]
                T.sync_threads()
            if tx == 0:
                tokens[slot] = T.if_then_else(active[slot] != 0, indices[0], -1)
                if active[slot] != 0:
                    positions[slot] += 1
        return (tokens, positions)

    return compute


def native_embedding(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c, vocab = (p["r"], p["c"], p["vocab"])

    @kernel
    def compute(
        ids: Tensor((r,), "int32"), w: Tensor((vocab, c), "bfloat16")
    ) -> Tensor((r, c), "bfloat16"):
        out = output((r, c), "bfloat16")
        with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                if col < c:
                    out[b, col] = T.if_then_else(
                        ids[b] >= 0, w[T.max(ids[b], 0), col], 0
                    )
        return out

    return compute


def argmax(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r, c = (p["r"], p["c"])
    maximum = tree_reduce(64, "max")

    @kernel
    def compute(
        x: Tensor((r * c,), "float32"), enabled: Tensor((r,), "int32")
    ) -> Tensor((r,), "int32"):
        out = output((r,), "int32", position=1)
        with T.Kernel(r, threads=64) as row:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((64,), "float32")
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            best = -T.infinity("float32")
            index = c
            for tile in T.serial(T.ceildiv(c, 64)):
                if tile * 64 + tx < c:
                    if x[row * c + tile * 64 + tx] > best:
                        best = x[row * c + tile * 64 + tx]
                        index = tile * 64 + tx
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            value = T.alloc_var("float32")
            value = scratch[0]
            T.sync_threads()
            scratch[tx] = T.cast(T.if_then_else(best == value, -index, -c), "float32")
            T.sync_threads()
            maximum(scratch, tx)
            if tx == 0 and enabled[row] != 0:
                out[row] = T.cast(-scratch[0], "int32")
        return out

    return compute


def embedding(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c = (p["r"], p["c"])
    v = p["v"]
    start = p.get("start", 0)

    @kernel
    def compute(
        tokens: Tensor((r,), "int32"),
        w: Tensor((v * c,), p.get("weight_dtype", "float32")),
    ) -> Tensor((r * c,), "float32"):
        out = output((r * c,), "float32")
        with T.Kernel(T.ceildiv(c, 64), r, threads=64) as (block, row):
            tx = T.get_thread_binding()
            col = block * 64 + tx
            if col < c and tokens[row] >= start and (tokens[row] < start + v):
                out[row * c + col] = w[(tokens[row] - start) * c + col]
        return out

    return compute


def embedding_packed(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.weights import bind

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    q, v = (p["type"], p["v"])
    weights = bind(p, (v, c), provider="cuda")
    if weights.storage.metadata:
        raise ValueError("this packed operation ABI requires embedded metadata")
    count, dtype = (weights.storage.values.shape[0], weights.storage.values.dtype)
    read_weight = weights.read

    @kernel
    def compute(tokens: Tensor((r,), "int32"), w: Tensor((count,), dtype)) -> Tensor(
        (r * c,), "float32"
    ):
        out = output((r * c,), "float32")
        with T.Kernel(T.ceildiv(c, 256), r, threads=256) as (bx, row):
            for lane in T.Parallel(256):
                col = bx * 256 + lane
                if col < c:
                    out[row * c + col] = read_weight((w,), tokens[row], col)
        return out

    return compute


def argmax_packed(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.vocabulary.kernel import _argmax as argmax_kernel

    p = {**spec.parameters, **schedule}
    return argmax_kernel(p)


def _argmax(p, explicit_unroll=False):
    import tilelang.language as T

    n, threads = (p["n"], 256)

    @kernel
    def compute(
        logits: Tensor((n,), "float32"), pos: Tensor((2,), "int32")
    ) -> tuple[Tensor((1,), "int32"), Tensor((2,), "int32")]:
        token = output((1,), "int32", position=1)
        if explicit_unroll:
            T.func_attr({"tensor.webgpu.loop_unroll": "explicit"})
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            values = T.alloc_shared((threads,), "float32")
            indices = T.alloc_shared((threads,), "int32")
            best = -T.infinity("float32")
            index = n
            for tile in T.serial(T.ceildiv(n, threads)):
                i = tile * threads + tx
                if i < n:
                    if (logits[i] > best) | (logits[i] == best) & (i < index):
                        best = logits[i]
                        index = i
            values[tx] = best
            indices[tx] = index
            T.sync_threads()
            for step in T.unroll(8):
                stride = 128 >> step
                if tx < stride:
                    if (values[tx + stride] > values[tx]) | (
                        values[tx + stride] == values[tx]
                    ) & (indices[tx + stride] < indices[tx]):
                        values[tx] = values[tx + stride]
                        indices[tx] = indices[tx + stride]
                T.sync_threads()
            if tx == 0:
                token[0] = indices[0]
                pos[1] = 1
        return (token, pos)

    return compute


def packed_word_argmax(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    r = p.get("r", 1)
    from akas.ops.vocabulary.kernel import _argmax as argmax_kernel

    return argmax_kernel(p, explicit_unroll=True)


def packed_word_embedding(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.weights import bind

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    c, q, v = (p["c"], p["type"], p["v"])
    weights = bind(p, (v, c), provider="webgpu")
    if weights.storage.metadata:
        raise ValueError("this packed operation ABI requires embedded metadata")
    count, dtype = (weights.storage.values.shape[0], weights.storage.values.dtype)
    decoder = weights.read

    @kernel
    def compute(tokens: Tensor((r,), "int32"), w: Tensor((count,), dtype)) -> Tensor(
        (r * c,), "float32"
    ):
        out = output((r * c,), "float32")
        with T.Kernel(T.ceildiv(c, 256), r, threads=256) as (bx, row):
            for lane in T.Parallel(256):
                col = bx * 256 + lane
                if col < c:
                    out[row * c + col] = decoder((w,), tokens[row], col)
        return out

    return compute


def argmax_partial(spec, schedule):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    p = {**spec.parameters, **schedule}
    rows, columns, block = p["r"], p["c"], p["block"]
    parts = (columns + block - 1) // block
    maximum = tree_reduce(128, "max")

    @kernel
    def compute(
        x: Tensor((rows * columns,), "float32"),
    ) -> tuple[Tensor((rows, parts), "float32"), Tensor((rows, parts), "int32")]:
        values = output((rows, parts), "float32")
        indices = output((rows, parts), "int32")
        with T.Kernel(parts, rows, threads=128) as (part, row):
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            best = -T.infinity("float32")
            index = columns
            scratch = T.alloc_shared((128,), "float32")
            winners = T.alloc_shared((128,), "int32")
            for tile in T.serial(T.ceildiv(block, 128)):
                col = part * block + tile * 128 + tx
                if col < columns and tile * 128 + tx < block:
                    if x[row * columns + col] > best:
                        best = x[row * columns + col]
                        index = col
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            peak = T.alloc_var("float32")
            peak = scratch[0]
            winners[tx] = T.if_then_else(best == peak, -index, -columns)
            T.sync_threads()
            maximum(winners, tx)
            if tx == 0:
                values[row, part] = peak
                indices[row, part] = -winners[0]
        return values, indices

    return compute


def argmax_merge(spec, schedule):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    p = {**spec.parameters, **schedule}
    rows, columns, block = p["r"], p["c"], p["block"]
    parts = (columns + block - 1) // block
    maximum = tree_reduce(128, "max")

    @kernel
    def compute(
        values: Tensor((rows, parts), "float32"),
        indices: Tensor((rows, parts), "int32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((rows,), "int32"):
        out = output((rows,), "int32")
        with T.Kernel(rows, threads=128) as row:
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            best = -T.infinity("float32")
            index = columns
            scratch = T.alloc_shared((128,), "float32")
            winners = T.alloc_shared((128,), "int32")
            for tile in T.serial(T.ceildiv(parts, 128)):
                part = tile * 128 + tx
                if part < parts:
                    value = values[row, part]
                    candidate = indices[row, part]
                    if value > best or (value == best and candidate < index):
                        best = value
                        index = candidate
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            peak = T.alloc_var("float32")
            peak = scratch[0]
            winners[tx] = T.if_then_else(best == peak, -index, -columns)
            T.sync_threads()
            maximum(winners, tx)
            if tx == 0 and enabled[row] != 0:
                out[row] = -winners[0]
        return out

    return compute
