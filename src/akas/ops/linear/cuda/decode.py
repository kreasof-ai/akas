"""Operation kernels for linear.cuda.decode."""


def make_kernel(kind, p):
    import tilelang.language as T

    if kind == "bf16_linear":
        r, k, o = p["r"], p["k"], p["o"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "bfloat16"),
            out: T.Tensor((r, o), "float32"),
        ):
            with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
                accum = T.alloc_fragment((r, 4, 32), "float32")
                total = T.alloc_fragment((r, 4), "float32")
                T.clear(accum)
                for tile in T.serial(k // 32):
                    for b, n, lane in T.Parallel(r, 4, 32):
                        if bx * 4 + n < o:
                            accum[b, n, lane] = T.ieee_fmaf(
                                T.cast(x[b, tile * 32 + lane], "float32"),
                                T.cast(w[bx * 4 + n, tile * 32 + lane], "float32"),
                                accum[b, n, lane],
                            )
                T.reduce_sum(accum, total, dim=2)
                for b, n in T.Parallel(r, 4):
                    if bx * 4 + n < o:
                        out[b, bx * 4 + n] = total[b, n]

        return kernel
    import tilelang.language as T

    if kind == "fp8_linear":
        r, k, o = p["r"], p["k"], p["o"]
        columns, threads = p.get("columns", 4), p.get("threads", 128)
        if (
            any(type(v) is not int or v <= 0 for v in (r, k, o, columns, threads))
            or k % 128
            or o % columns
            or threads % 32
            or threads > 1024
            or r > 8
            or columns * 32 % threads
        ):
            raise ValueError("invalid native FP8 GEMV schedule")

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "uint8"),
            scales: T.Tensor((T.ceildiv(o, 128), k // 128), "bfloat16"),
            out: T.Tensor((r, o), "float32"),
        ):
            with T.Kernel(o // columns, threads=threads) as bx:
                accum = T.alloc_fragment((r, columns, 32), "float32")
                total = T.alloc_fragment((r, columns), "float32")
                coefficient = T.alloc_fragment((columns, 32), "float32")
                activation = T.alloc_shared((r, 128), "float32")
                absolute = T.alloc_fragment((r, 128), "float32")
                maximum = T.alloc_fragment((r,), "float32")
                T.clear(accum)
                for tile in T.serial(k // 128):
                    for b, j in T.Parallel(r, 128):
                        absolute[b, j] = T.abs(T.cast(x[b, tile * 128 + j], "float32"))
                    T.reduce_max(absolute, maximum, dim=1)
                    for b, j in T.Parallel(r, 128):
                        scale = T.max(maximum[b], T.float32(1e-12)) / T.float32(448)
                        encoded = T.call_extern(
                            "uint32",
                            "tensor_encode_e4m3",
                            T.cast(x[b, tile * 128 + j], "float32") / scale,
                        )
                        activation[b, j] = (
                            T.call_extern("float32", "tensor_decode_e4m3", encoded)
                            * scale
                        )
                    for part in T.unroll(4):
                        for n, lane in T.Parallel(columns, 32):
                            coefficient[n, lane] = T.call_extern(
                                "float32",
                                "tensor_decode_e4m3",
                                T.cast(
                                    w[bx * columns + n, tile * 128 + part * 32 + lane],
                                    "uint32",
                                ),
                            )
                            coefficient[n, lane] *= T.cast(
                                scales[(bx * columns + n) // 128, tile], "float32"
                            )
                        for b, n, lane in T.Parallel(r, columns, 32):
                            accum[b, n, lane] = T.ieee_fmaf(
                                T.cast(activation[b, part * 32 + lane], "float32"),
                                coefficient[n, lane],
                                accum[b, n, lane],
                            )
                T.reduce_sum(accum, total, dim=2)
                for b, n in T.Parallel(r, columns):
                    out[b, bx * columns + n] = total[b, n]

        return kernel
    raise ValueError("unsupported operation: " + kind)
