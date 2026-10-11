"""Shared normalization and activations; geometry and precision are parameters."""


def swiglu_kernel(p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, c = p["r"], p["c"]
    flat, packed = p.get("flat", False), p.get("packed", False)
    rounded = p.get("round_inputs", False)
    input_dtype = p.get("input_dtype", "float32")
    output_dtype = p.get("output_dtype", input_dtype)
    round_dtype = p.get("round_dtype", output_dtype)
    input_shape = (r * c,) if flat else (r, (2 if packed else 1) * c)
    output_shape = (r * c,) if flat else (r, c)
    if flat and packed:
        raise ValueError("packed SwiGLU uses a two-dimensional projection")

    @T.macro
    def element(g, u, out, row, column):
        gate = T.alloc_var("float32")
        up = T.alloc_var("float32")
        if flat:
            gate = T.cast(g[row * c + column], "float32")
            up = T.cast(u[row * c + column], "float32")
        else:
            gate = T.cast(g[row, column], "float32")
            if packed:
                up = T.cast(g[row, c + column], "float32")
            else:
                up = T.cast(u[row, column], "float32")
        if rounded:
            gate = T.cast(T.cast(gate, round_dtype), "float32")
            up = T.cast(T.cast(up, round_dtype), "float32")
        if flat:
            out[row * c + column] = gate / (1 + T.exp(-gate)) * up
        else:
            out[row, column] = gate / (1 + T.exp(-gate)) * up

    @T.macro
    def separate(g, u, out):
        if flat:
            with T.Kernel(T.ceildiv(r * c, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < r * c:
                        element(g, u, out, i // c, i % c)
        else:
            with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (row, tile):
                for lane in T.Parallel(256):
                    column = tile * 256 + lane
                    if column < c:
                        element(g, u, out, row, column)

    @T.macro
    def joined(x, out):
        separate(x, x, out)

    names = p.get("input_names", ("g", "u"))
    arguments = [("x" if packed else names[0], input_shape, input_dtype)]
    if not packed:
        arguments.append((names[1], input_shape, input_dtype))
    arguments.append(("out", output_shape, output_dtype))
    return primitive(arguments, joined if packed else separate)
