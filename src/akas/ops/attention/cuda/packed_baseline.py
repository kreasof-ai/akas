"""Operation kernels for attention.cuda.packed_baseline."""

import math


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "qkv":
        h, kh, d, cap = (p["h"], p["kh"], p["d"], p["cap"])
        eps, theta = (p["eps"], p["theta"])

        @T.prim_func
        def kernel(
            q: T.Tensor((r * h * d,), "float32"),
            k: T.Tensor((r * kh * d,), "float32"),
            v: T.Tensor((r * kh * d,), "float32"),
            qw: T.Tensor((d,), "float32"),
            kw: T.Tensor((d,), "float32"),
            qo: T.Tensor((r * h * d,), "float32"),
            kc: T.Tensor((cap * kh * d,), "float16"),
            vc: T.Tensor((cap * kh * d,), "float16"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(h, r, threads=64) as (head, row):
                square = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                norm = T.alloc_shared((d,), "float32")
                for i in T.Parallel(d):
                    square[i] = (
                        q[(row * h + head) * d + i] * q[(row * h + head) * d + i]
                    )
                T.reduce_sum(square, total, dim=0)
                for i in T.Parallel(d):
                    norm[i] = (
                        q[(row * h + head) * d + i]
                        * T.rsqrt(total[0] / d + eps)
                        * qw[i]
                    )
                for i in T.Parallel(d // 2):
                    angle = T.cast(pos[0] + row, "float32") * T.exp(
                        -(math.log(theta) * 2 / d) * i
                    )
                    qo[(row * h + head) * d + i] = norm[i] * T.cos(angle) - norm[
                        i + d // 2
                    ] * T.sin(angle)
                    qo[(row * h + head) * d + i + d // 2] = norm[i] * T.sin(
                        angle
                    ) + norm[i + d // 2] * T.cos(angle)
                if head < kh:
                    for i in T.Parallel(d):
                        square[i] = (
                            k[(row * kh + head) * d + i] * k[(row * kh + head) * d + i]
                        )
                    T.reduce_sum(square, total, dim=0)
                    for i in T.Parallel(d):
                        norm[i] = (
                            k[(row * kh + head) * d + i]
                            * T.rsqrt(total[0] / d + eps)
                            * kw[i]
                        )
                        if row < pos[1]:
                            vc[((pos[0] + row) * kh + head) * d + i] = v[
                                (row * kh + head) * d + i
                            ]
                    for i in T.Parallel(d // 2):
                        angle = T.cast(pos[0] + row, "float32") * T.exp(
                            -(math.log(theta) * 2 / d) * i
                        )
                        if row < pos[1]:
                            kc[((pos[0] + row) * kh + head) * d + i] = norm[i] * T.cos(
                                angle
                            ) - norm[i + d // 2] * T.sin(angle)
                            kc[((pos[0] + row) * kh + head) * d + i + d // 2] = norm[
                                i
                            ] * T.sin(angle) + norm[i + d // 2] * T.cos(angle)

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "attention":
        h, kh, d, cap = (p["h"], p["kh"], p["d"], p["cap"])
        if r == 1:

            @T.prim_func
            def kernel(
                q: T.Tensor((r * h * d,), "float32"),
                kc: T.Tensor((cap * kh * d,), "float16"),
                vc: T.Tensor((cap * kh * d,), "float16"),
                out: T.Tensor((r * h * d,), "float32"),
                pos: T.Tensor((2,), "int32"),
            ):
                with T.Kernel(h, threads=128) as head:
                    dots = T.alloc_fragment((64, d), "float32")
                    scores = T.alloc_fragment((64,), "float32")
                    products = T.alloc_fragment((64, d), "float32")
                    partial = T.alloc_fragment((d,), "float32")
                    result = T.alloc_fragment((d,), "float32")
                    maximum = T.alloc_fragment((1,), "float32")
                    previous = T.alloc_fragment((1,), "float32")
                    normalizer = T.alloc_fragment((1,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    T.fill(maximum, -T.infinity("float32"))
                    T.clear(result)
                    T.clear(normalizer)
                    for tile in T.serial(T.ceildiv(pos[0] + 1, 64)):
                        for i, j in T.Parallel(64, d):
                            dots[i, j] = q[head * d + j] * T.cast(
                                kc[((tile * 64 + i) * kh + head // (h // kh)) * d + j],
                                "float32",
                            )
                        T.reduce_sum(dots, scores, dim=1)
                        T.copy(maximum, previous)
                        for i in T.Parallel(64):
                            scores[i] = T.if_then_else(
                                tile * 64 + i <= pos[0],
                                scores[i] * d ** (-0.5),
                                -T.infinity("float32"),
                            )
                        T.reduce_max(scores, maximum, dim=0, clear=False)
                        for i in T.Parallel(64):
                            scores[i] = T.exp(scores[i] - maximum[0])
                        T.reduce_sum(scores, total, dim=0)
                        normalizer[0] = (
                            normalizer[0] * T.exp(previous[0] - maximum[0]) + total[0]
                        )
                        for i, j in T.Parallel(64, d):
                            products[i, j] = scores[i] * T.cast(
                                vc[((tile * 64 + i) * kh + head // (h // kh)) * d + j],
                                "float32",
                            )
                        T.reduce_sum(products, partial, dim=0)
                        for j in T.Parallel(d):
                            result[j] = (
                                result[j] * T.exp(previous[0] - maximum[0]) + partial[j]
                            )
                    for j in T.Parallel(d):
                        out[head * d + j] = result[j] / normalizer[0]

            return kernel

        @T.prim_func
        def kernel(
            q: T.Tensor((r * h * d,), "float32"),
            kc: T.Tensor((cap * kh * d,), "float16"),
            vc: T.Tensor((cap * kh * d,), "float16"),
            out: T.Tensor((r * h * d,), "float32"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(r, 32), h, threads=128) as (bx, head):
                query = T.alloc_shared((32, d), "float16")
                key = T.alloc_shared((64, d), "float16")
                value = T.alloc_shared((64, d), "float16")
                prob = T.alloc_shared((32, 64), "float16")
                scores = T.alloc_fragment((32, 64), "float32")
                result = T.alloc_fragment((32, d), "float32")
                maximum = T.alloc_fragment((32,), "float32")
                previous = T.alloc_fragment((32,), "float32")
                factor = T.alloc_fragment((32,), "float32")
                normalizer = T.alloc_fragment((32,), "float32")
                total = T.alloc_fragment((32,), "float32")
                for i, j in T.Parallel(32, d):
                    query[i, j] = T.if_then_else(
                        bx * 32 + i < r, q[((bx * 32 + i) * h + head) * d + j], 0
                    )
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                for tile in T.serial(T.ceildiv(pos[0] + T.min(r, (bx + 1) * 32), 64)):
                    for i, j in T.Parallel(64, d):
                        key[i, j] = kc[
                            ((tile * 64 + i) * kh + head // (h // kh)) * d + j
                        ]
                        value[i, j] = vc[
                            ((tile * 64 + i) * kh + head // (h // kh)) * d + j
                        ]
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(32, 64):
                        scores[i, j] = T.if_then_else(
                            tile * 64 + j <= pos[0] + bx * 32 + i,
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(32):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(32, 64):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(32):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(32, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
                for i, j in T.Parallel(32, d):
                    if bx * 32 + i < r:
                        out[((bx * 32 + i) * h + head) * d + j] = T.if_then_else(
                            bx * 32 + i < pos[1], result[i, j] / normalizer[i], 0
                        )

        return kernel
    raise ValueError("unsupported operation: " + kind)
