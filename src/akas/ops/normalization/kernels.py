"""Shared normalization and activations; geometry and precision are parameters."""


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
