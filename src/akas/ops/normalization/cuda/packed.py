"""Packed projections, attention, normalization, and reductions for CUDA.

Factories are producer-only; consumers execute Tensor artifacts."""

from akas.ops.normalization.kernels import rms_kernel


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


def make_kernel(kind, p):
    if kind in ("qnorm", "kvnorm"):
        return normalization_kernel(kind, p)
    if kind == "add_rms":
        return rms_kernel(dict(p, flat=True, fused=True, separate_residual=True))
    raise ValueError("unsupported operation: " + kind)
