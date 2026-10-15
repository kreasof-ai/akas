"""Named kernel factories and shared implementation helpers."""


def projection_algorithm(p, *, minimum_groups=0, default_tile=(16, 32, 64)):
    algorithm = p.get("algorithm", "auto")
    if algorithm not in ("auto", "gemv", "mma"):
        raise ValueError("unknown floating projection algorithm")
    if algorithm != "auto":
        return algorithm
    m, n = p.get("tile", default_tile)[:2]
    m, n = (p.get("block_m", m), p.get("block_n", n))
    if p.get("schedule") == "outer":
        m, n = (
            p.get("outer", {}).get("tile_m", 64),
            p.get("outer", {}).get("tile_n", 64),
        )
    groups = (p["r"] + m - 1) // m * ((p["o"] + n - 1) // n)
    underfilled = groups < minimum_groups and p["r"] <= p.get(
        "maximum_gemv_rows", p["r"]
    )
    return "gemv" if p["r"] <= 8 or underfilled else "mma"


def batch_rows(kernel, rows, k, o):
    """Lift GEMV over rows, interleaving CTAs to reuse each weight tile."""
    import tvm
    from tvm import tirx as ir

    old_x = kernel.buffer_map[kernel.params[0]]
    old_out = kernel.buffer_map[kernel.params[-1]]
    x = ir.decl_buffer((rows * k,), old_x.dtype, old_x.name, data=old_x.data)
    out = ir.decl_buffer((rows * o,), old_out.dtype, old_out.name, data=old_out.data)

    def lift(node):
        if (
            not isinstance(node, ir.For)
            or node.thread_binding is None
            or node.thread_binding.thread_tag != "blockIdx.x"
        ):
            return None
        block = node.loop_var
        row = block % rows
        body = ir.stmt_functor.substitute(node.body, {block: block // rows})

        def rewrite(value):
            if isinstance(value, ir.BufferLoad) and value.buffer.same_as(old_x):
                return ir.BufferLoad(x, [row * k + value.indices[0]])
            if isinstance(value, ir.BufferStore) and value.buffer.same_as(old_out):
                return ir.BufferStore(out, value.value, [row * o + value.indices[0]])
            return None

        body = ir.stmt_functor.ir_transform(
            body, None, rewrite, ["tirx.BufferStore", "tirx.BufferLoad"]
        )
        binding = ir.IterVar(
            tvm.ir.Range(0, node.extent * rows),
            block,
            ir.IterVar.ThreadIndex,
            "blockIdx.x",
        )
        return ir.For(
            block, node.min, node.extent * rows, node.kind, body, thread_binding=binding
        )

    body = ir.stmt_functor.ir_transform(kernel.body, None, lift, ["tirx.For"])
    buffers = dict(kernel.buffer_map)
    buffers[kernel.params[0]], buffers[kernel.params[-1]] = (x, out)
    return ir.PrimFunc(kernel.params, body, kernel.ret_type, buffers, kernel.attrs)


def column_tiles(kernel):
    """Visit output columns first so large row batches reuse activation tiles."""
    from tvm import tirx as ir

    def rewrite(node):
        if not isinstance(node, ir.For) or node.thread_binding is None:
            return None
        tag = node.thread_binding.thread_tag
        if tag not in ("blockIdx.x", "blockIdx.y"):
            return None
        binding = ir.IterVar(
            node.thread_binding.dom,
            node.loop_var,
            ir.IterVar.ThreadIndex,
            "blockIdx.y" if tag == "blockIdx.x" else "blockIdx.x",
        )
        return ir.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            node.body,
            thread_binding=binding,
        )

    body = ir.stmt_functor.ir_transform(kernel.body, None, rewrite, ["tirx.For"])
    return ir.PrimFunc(
        kernel.params, body, kernel.ret_type, kernel.buffer_map, kernel.attrs
    )


def output_view(kernel, p):
    from tvm import tirx as ir

    old = kernel.buffer_map[kernel.params[-1]]
    r, o = (p["r"], p["o"])
    stride, offset = (p.get("output_stride", o), p.get("output_offset", 0))
    matrix = p.get("matrix_output", False)
    if not matrix and stride == o and (offset == 0):
        return kernel
    shape = (r, stride) if matrix else (r * stride,)
    new = ir.decl_buffer(shape, old.dtype, old.name, data=old.data)

    def indices(index):
        row, col = (index // o, index % o + offset)
        return [row, col] if matrix else [row * stride + col]

    def rewrite(node):
        if isinstance(node, ir.BufferStore) and node.buffer.same_as(old):
            return ir.BufferStore(new, node.value, indices(node.indices[0]))
        if isinstance(node, ir.BufferLoad) and node.buffer.same_as(old):
            return ir.BufferLoad(new, indices(node.indices[0]))
        return None

    body = ir.stmt_functor.ir_transform(
        kernel.body, None, rewrite, ["tirx.BufferStore", "tirx.BufferLoad"]
    )
    buffers = dict(kernel.buffer_map)
    buffers[kernel.params[-1]] = new
    return ir.PrimFunc(kernel.params, body, kernel.ret_type, buffers, kernel.attrs)
