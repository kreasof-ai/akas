"""TileLang DSL for packed portable and standard-subgroup LFM2 schedules."""

import math
from akas.ops.reductions.webgpu import tree_reduce
from akas.ops.reductions.webgpu import subgroup_reduce
from akas.ops.quantization.webgpu.storage import round_half


def rms_width(c):
    for width in (256, 128, 64):
        if c % width == 0:
            return width
    return 64


def rms_kernel(kind, p):
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


def normalization_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, h, kh, d, cap = p.get("r", 1), p["h"], p["kh"], p["d"], p["cap"]
    query = kind == "qnorm"
    heads, eps, angle_scale = h if query else kh, p["eps"], math.log(p["theta"]) * 2 / d
    rounded = round_half()

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
            if query:
                for i in T.Parallel(d // 2):
                    angle = T.cast(pos[0] + row, "float32") * T.exp(-angle_scale * i)
                    out[(row * heads + head) * d + i] = norm[i] * T.cos(angle) - norm[
                        i + d // 2
                    ] * T.sin(angle)
                    out[(row * heads + head) * d + i + d // 2] = norm[i] * T.sin(
                        angle
                    ) + norm[i + d // 2] * T.cos(angle)
            else:
                if row < pos[1]:
                    for i in T.Parallel(d // 2):
                        angle = T.cast(pos[0] + row, "float32") * T.exp(
                            -angle_scale * i
                        )
                        out[((pos[0] + row) * heads + head) * d + i] = rounded(
                            norm[i] * T.cos(angle) - norm[i + d // 2] * T.sin(angle)
                        )
                        out[((pos[0] + row) * heads + head) * d + i + d // 2] = rounded(
                            norm[i] * T.sin(angle) + norm[i + d // 2] * T.cos(angle)
                        )
                    for i in T.Parallel(d):
                        vc[((pos[0] + row) * heads + head) * d + i] = rounded(
                            v[(row * heads + head) * d + i]
                        )

    @T.macro
    def call(k, v, kw, kc, vc, pos):
        normalize(k, kw, kc, pos, v, vc)

    if query:
        return primitive(
            [
                ("q", r * h * d, "float32"),
                ("qw", d, "float32"),
                ("qo", r * h * d, "float32"),
                ("pos", 2, "int32"),
            ],
            normalize,
        )
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


def make_kernel(kind, p):
    r = p.get("r", 1)
    if kind == "rms" and (r == 1 or p.get("parallel_rows")) or kind == "add_rms":
        return rms_kernel(kind, p)
    if kind in ("qnorm", "kvnorm"):
        return normalization_kernel(kind, p)
    raise ValueError("unsupported operation: " + kind)
