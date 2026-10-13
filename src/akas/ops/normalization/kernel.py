"""Normalization kernels with explicit layout, precision and reduction policies."""

from akas.ops.reductions.webgpu import tree_reduce, subgroup_reduce


def rms_kernel(p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, c, eps = p["r"], p["c"], p["eps"]
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

    arguments = [(p.get("input_name", "x"), shape, source_dtype)]
    if fused:
        if separate:
            arguments.append(("mixed", shape, source_dtype))
        arguments.append(("residual", shape, residual_dtype))
    arguments += [("w", (c,), weight_dtype), ("out", shape, output_dtype)]
    algorithm = separate_output if separate else in_place if fused else ordinary
    return primitive(arguments, algorithm)


def rms_width(c):
    for width in (256, 128, 64):
        if c % width == 0:
            return width
    return 64


def parallel_rms_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, c, eps = p.get("r", 1), p["c"], p["eps"]
    threads, fused = rms_width(c), kind == "add_rms"
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

        @T.macro
        def call(x, mixed, residual, w, out):
            normalize(x, w, out, mixed, residual)

        return primitive(
            [
                ("x", r * c, "float32"),
                ("mixed", r * c, "float32"),
                ("residual", r * c, "float32"),
                ("w", c, "float32"),
                ("out", r * c, "float32"),
            ],
            call,
        )
    return primitive(
        [("x", r * c, "float32"), ("w", c, "float32"), ("out", r * c, "float32")],
        normalize,
    )


def serial_rms_kernel(kind, p):
    import tilelang.language as T
    from akas.ops.reductions.webgpu import tree_reduce

    r, c, eps = p["r"], p["c"], p["eps"]
    reduce = tree_reduce(64)

    @T.prim_func
    def kernel(
        x: T.Tensor((r * c,), "float32"),
        w: T.Tensor((c,), "float32"),
        out: T.Tensor((r * c,), "float32"),
    ):
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

    return kernel


def normalization_kernel(kind, p):
    import math
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "cap"))
    heads = h if kind == "qnorm" else kh
    eps, angle_scale = p["eps"], math.log(p["theta"]) * 2 / d

    @T.macro
    def normalize(x, w, out, pos, v=None, vc=None):
        with T.Kernel(heads, r, threads=64) as (head, row):
            square = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            norm = T.alloc_shared((d,), "float32")
            for i in T.Parallel(d):
                square[i] = (
                    x[(row * heads + head) * d + i] * x[(row * heads + head) * d + i]
                )
            T.reduce_sum(square, total, dim=0)
            for i in T.Parallel(d):
                norm[i] = (
                    x[(row * heads + head) * d + i] * T.rsqrt(total[0] / d + eps) * w[i]
                )
                if kind == "kvnorm":
                    if row < pos[1]:
                        vc[((pos[0] + row) * kh + head) * d + i] = v[
                            (row * kh + head) * d + i
                        ]
            for i in T.Parallel(d // 2):
                angle = T.cast(pos[0] + row, "float32") * T.exp(-angle_scale * i)
                if kind == "qnorm":
                    out[(row * heads + head) * d + i] = norm[i] * T.cos(angle) - norm[
                        i + d // 2
                    ] * T.sin(angle)
                    out[(row * heads + head) * d + i + d // 2] = norm[i] * T.sin(
                        angle
                    ) + norm[i + d // 2] * T.cos(angle)
                else:
                    if row < pos[1]:
                        out[((pos[0] + row) * heads + head) * d + i] = norm[i] * T.cos(
                            angle
                        ) - norm[i + d // 2] * T.sin(angle)
                        out[((pos[0] + row) * heads + head) * d + i + d // 2] = norm[
                            i
                        ] * T.sin(angle) + norm[i + d // 2] * T.cos(angle)

    if kind == "qnorm":
        return primitive(
            [
                ("q", r * h * d, "float32"),
                ("qw", d, "float32"),
                ("qo", r * h * d, "float32"),
                ("pos", 2, "int32"),
            ],
            normalize,
        )

    @T.macro
    def call(k, v, kw, kc, vc, pos):
        normalize(k, kw, kc, pos, v, vc)

    return primitive(
        [
            ("k", r * kh * d, "float32"),
            ("v", r * kh * d, "float32"),
            ("kw", d, "float32"),
            ("kc", cap * kh * d, "float16"),
            ("vc", cap * kh * d, "float16"),
            ("pos", 2, "int32"),
        ],
        call,
    )


def make_kernel(kind, parameters):
    p = dict(parameters)
    algorithm = p.pop("algorithm", "fragment")
    if kind in ("qnorm", "kvnorm"):
        return normalization_kernel(kind, p)
    if kind not in ("rms", "add_rms"):
        raise ValueError("unsupported normalization: " + kind)
    if algorithm == "serial-tree":
        return serial_rms_kernel(kind, p)
    if algorithm == "parallel-tree":
        return parallel_rms_kernel(kind, p)
    if algorithm != "fragment":
        raise ValueError("unknown normalization reduction: " + algorithm)
    return rms_kernel({**p, "fused": kind == "add_rms"})
