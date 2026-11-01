"""Shared TileLang computations for normalization."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def add_rms(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.normalization.kernel import _fragment as rms_kernel
    from akas.ops.normalization.kernel import _parallel as parallel_rms_kernel
    from akas.ops.normalization.kernel import _serial as serial_rms_kernel

    parameters = {**spec.parameters, **schedule}
    p = dict(parameters)
    algorithm = p.pop("algorithm", "fragment")
    if algorithm == "serial-tree":
        return serial_rms_kernel(p)
    if algorithm == "parallel-tree":
        return parallel_rms_kernel(p, fused=True)
    if algorithm != "fragment":
        raise ValueError("unknown normalization reduction: " + algorithm)
    return rms_kernel({**p, "fused": True})


def kvnorm(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.kernel import _normalization as normalization_kernel

    parameters = {**spec.parameters, **schedule}
    p = dict(parameters)
    algorithm = p.pop("algorithm", "fragment")
    return normalization_kernel(p, query=False)


def qnorm(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.kernel import _normalization as normalization_kernel

    parameters = {**spec.parameters, **schedule}
    p = dict(parameters)
    algorithm = p.pop("algorithm", "fragment")
    return normalization_kernel(p, query=True)


def rms(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.normalization.kernel import _fragment as rms_kernel
    from akas.ops.normalization.kernel import _parallel as parallel_rms_kernel
    from akas.ops.normalization.kernel import _serial as serial_rms_kernel

    parameters = {**spec.parameters, **schedule}
    p = dict(parameters)
    algorithm = p.pop("algorithm", "fragment")
    if algorithm == "serial-tree":
        return serial_rms_kernel(p)
    if algorithm == "parallel-tree":
        return parallel_rms_kernel(p, fused=False)
    if algorithm != "fragment":
        raise ValueError("unknown normalization reduction: " + algorithm)
    return rms_kernel({**p, "fused": False})


def _fragment(p):
    import tilelang.language as T

    r, c, eps = (p["r"], p["c"], p["eps"])
    flat = p.get("flat", False)
    fused = p.get("fused", False)
    separate = p.get("separate_residual", False)
    centered = p.get("zero_centered", False)
    rounded = p.get("round_inputs", False)
    source_dtype = p.get("input_dtype", "float32")
    residual_dtype = p.get("residual_dtype", source_dtype)
    output_dtype = p.get("output_dtype", source_dtype)
    weight_dtype = p.get("weight_dtype", source_dtype)
    shape = (r * c,) if flat else (r, c)

    @T.macro
    def load(buffer, row, column):
        if flat:
            return T.cast(buffer[row * c + column], "float32")
        else:
            return T.cast(buffer[row, column], "float32")

    @T.macro
    def store(buffer, row, column, value):
        if flat:
            buffer[row * c + column] = value
        else:
            buffer[row, column] = value

    @T.macro
    def normalize(x, other, residual, w, out):
        with T.Kernel(r, threads=256) as row:
            values = T.alloc_fragment((c,), "float32")
            squares = T.alloc_fragment((c,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(c):
                if fused:
                    if rounded:
                        values[j] = T.cast(
                            T.cast(T.cast(load(x, row, j), residual_dtype), "float32")
                            + load(other, row, j),
                            residual_dtype,
                        )
                    else:
                        values[j] = T.cast(
                            load(x, row, j) + load(other, row, j), residual_dtype
                        )
                else:
                    values[j] = load(x, row, j)
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(c):
                if fused:
                    store(residual, row, j, values[j])
                if centered:
                    store(
                        out,
                        row,
                        j,
                        values[j]
                        * T.rsqrt(total[0] / c + eps)
                        * (1 + T.cast(w[j], "float32")),
                    )
                else:
                    store(
                        out,
                        row,
                        j,
                        values[j]
                        * T.rsqrt(total[0] / c + eps)
                        * T.cast(w[j], "float32"),
                    )

    @T.macro
    def ordinary(x, w, out):
        normalize(x, x, x, w, out)

    @T.macro
    def in_place(x, residual, w, out):
        normalize(x, residual, residual, w, out)

    @T.macro
    def separate_output(x, mixed, residual, w, out):
        normalize(x, mixed, residual, w, out)

    algorithm = separate_output if separate else in_place if fused else ordinary
    if fused and separate:

        @kernel(names={"x": p.get("input_name", "x")})
        def compute(
            x: Tensor(shape, source_dtype),
            mixed: Tensor(shape, source_dtype),
            w: Tensor((c,), weight_dtype),
        ) -> tuple[Tensor(shape, residual_dtype), Tensor(shape, output_dtype)]:
            residual = output(shape, residual_dtype, position=2)
            out = output(shape, output_dtype, position=4)
            normalize(x, mixed, residual, w, out)
            return (residual, out)

        return compute
    if fused and (not separate):

        @kernel(names={"x": p.get("input_name", "x")})
        def compute(
            x: Tensor(shape, source_dtype),
            residual: Tensor(shape, residual_dtype),
            w: Tensor((c,), weight_dtype),
        ) -> tuple[Tensor(shape, residual_dtype), Tensor(shape, output_dtype)]:
            out = output(shape, output_dtype)
            normalize(x, residual, residual, w, out)
            return (residual, out)

        return compute
    if not fused and (not separate):

        @kernel(names={"x": p.get("input_name", "x")})
        def compute(
            x: Tensor(shape, source_dtype), w: Tensor((c,), weight_dtype)
        ) -> Tensor(shape, output_dtype):
            out = output(shape, output_dtype)
            normalize(x, x, x, w, out)
            return out

        return compute


def rms_width(c):
    for width in (256, 128, 64):
        if c % width == 0:
            return width
    return 64


def _parallel(p, fused=False):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce
    from akas.kernels.webgpu import subgroup_reduce

    r, c, eps = (p.get("r", 1), p["c"], p["eps"])
    threads = rms_width(c)
    reduce = subgroup_reduce(threads) if p.get("sg") else tree_reduce(threads)

    @T.macro
    def value(x, mixed, index):
        if fused:
            return x[index] + mixed[index]
        else:
            return x[index]

    @T.macro
    def normalize(x, w, out, mixed=None, residual=None):
        with T.Kernel(r, threads=threads) as row:
            tx = T.get_thread_binding()
            total = T.alloc_var("float32")
            scratch = T.alloc_shared((threads,), "float32")
            total = 0
            for i in T.serial(c // threads):
                total = total + value(x, mixed, row * c + i * threads + tx) * value(
                    x, mixed, row * c + i * threads + tx
                )
            if p.get("sg"):
                reduce(scratch, tx, total)
            else:
                scratch[tx] = total
                T.sync_threads()
                reduce(scratch, tx)
            for i in T.serial(c // threads):
                if fused:
                    residual[row * c + i * threads + tx] = (
                        x[row * c + i * threads + tx]
                        + mixed[row * c + i * threads + tx]
                    )
                out[row * c + i * threads + tx] = (
                    value(x, mixed, row * c + i * threads + tx)
                    * T.rsqrt(scratch[0] / c + eps)
                    * w[i * threads + tx]
                )

    if fused:

        @kernel
        def compute(
            x: Tensor(r * c, "float32"),
            mixed: Tensor(r * c, "float32"),
            residual: Tensor(r * c, "float32"),
            w: Tensor(c, "float32"),
        ) -> tuple[Tensor(r * c, "float32"), Tensor(r * c, "float32")]:
            out = output(r * c, "float32")
            normalize(x, w, out, mixed, residual)
            return (residual, out)

        return compute

    @kernel
    def compute(x: Tensor(r * c, "float32"), w: Tensor(c, "float32")) -> Tensor(
        r * c, "float32"
    ):
        out = output(r * c, "float32")
        normalize(x, w, out)
        return out

    return compute


def _serial(p):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r, c, eps = (p["r"], p["c"], p["eps"])
    reduce = tree_reduce(64)

    @kernel
    def compute(x: Tensor((r * c,), "float32"), w: Tensor((c,), "float32")) -> Tensor(
        (r * c,), "float32"
    ):
        out = output((r * c,), "float32")
        with T.Kernel(r, threads=64) as row:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((64,), "float32")
            acc = T.alloc_var("float32")
            acc = 0
            for tile in T.serial(T.ceildiv(c, 64)):
                if tile * 64 + tx < c:
                    acc += x[row * c + tile * 64 + tx] * x[row * c + tile * 64 + tx]
            scratch[tx] = acc
            T.sync_threads()
            reduce(scratch, tx)
            for tile in T.serial(T.ceildiv(c, 64)):
                if tile * 64 + tx < c:
                    out[row * c + tile * 64 + tx] = (
                        x[row * c + tile * 64 + tx]
                        * T.rsqrt(scratch[0] / c + eps)
                        * w[tile * 64 + tx]
                    )
        return out

    return compute
