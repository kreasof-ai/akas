"""Lift row-local kernel state into a persistent request-slot pool."""

from math import prod


def map_state(kernel, states, *, rows, capacity):
    """Map specified state parameters through one logical-row to request map.

    Arithmetic and reduction order are unchanged. Kernels already accepting a
    request mapping need no lift. State regions must remain inside one request;
    a copy spanning multiple request rows cannot be translated as one pointer.
    """
    import tvm
    from tvm import tirx as ir

    if rows < 1 or capacity < rows:
        raise ValueError("state pool must contain every execution row")
    selected = {}
    buffers = dict(kernel.buffer_map)
    for index in states:
        param = kernel.params[index]
        old = buffers[param]
        shape = tuple(int(value) for value in old.shape)
        if shape[0] % rows:
            raise ValueError("state must have a whole row-major request dimension")
        stride = prod(shape) // rows
        shape = (shape[0] // rows * capacity, *shape[1:])
        new = ir.decl_buffer(shape, old.dtype, old.name, data=old.data)
        selected[old.data] = old, new, stride
        buffers[param] = new
    mapping = ir.decl_buffer((rows,), "int32", "mapping")
    buffers[mapping.data] = mapping
    analyzer = tvm.arith.Analyzer()

    def bind_ranges(node):
        if isinstance(node, ir.For):
            analyzer.bind(node.loop_var, tvm.ir.Range.from_min_extent(node.min, node.extent))
        elif isinstance(node, ir.AttrStmt) and node.attr_key == "thread_extent":
            analyzer.bind(node.node.var, tvm.ir.Range.from_min_extent(0, node.value))

    ir.stmt_functor.post_order_visit(kernel.body, bind_ranges)

    def validate_region(node):
        if not isinstance(node, ir.Call):
            return None
        name = getattr(node.op, "name", "")
        if name == "tl.region" and isinstance(node.args[0], ir.BufferLoad):
            load = node.args[0]
            row = selected.get(load.buffer.data)
            if row is not None:
                old, _, _ = row
                span = int(old.shape[0]) // rows
                extent = node.args[2]
                if not analyzer.can_prove(load.indices[0] % span + extent <= span):
                    raise ValueError("state copy region crosses request rows")
        elif name in ("tirx.tvm_access_ptr", "tir.tvm_access_ptr"):
            row = selected.get(node.args[1])
            if row is not None:
                stride = row[2]
                if not analyzer.can_prove(node.args[2] % stride + node.args[3] <= stride):
                    raise ValueError("state pointer region crosses request rows")
        return None

    def address(index, stride):
        # Inactive rows may still read controls before their operation mask.
        # Their -1 sentinel must never become an out-of-bounds pointer.
        return ir.max(mapping[index // stride], 0) * stride + index % stride

    def indices(values, old, new, stride):
        flat = sum(
            value * prod(tuple(int(d) for d in old.shape[i + 1 :]))
            for i, value in enumerate(values)
        )
        flat = address(flat, stride)
        shape = tuple(int(d) for d in new.shape)
        return [(flat // prod(shape[i + 1 :])) % shape[i] for i in range(len(shape))]

    def rewrite(node):
        if isinstance(node, (ir.BufferLoad, ir.BufferStore)):
            row = selected.get(node.buffer.data)
            if row is None:
                return None
            old, new, stride = row
            index = indices(node.indices, old, new, stride)
            if isinstance(node, ir.BufferLoad):
                return ir.BufferLoad(new, index)
            return ir.BufferStore(new, node.value, index)
        if isinstance(node, ir.Call) and getattr(node.op, "name", "") in (
            "tirx.tvm_access_ptr",
            "tir.tvm_access_ptr",
        ):
            row = selected.get(node.args[1])
            if row is not None:
                _, _, stride = row
                return ir.Call(
                    node.dtype,
                    node.op,
                    [*node.args[:2], address(node.args[2], stride), *node.args[3:]],
                )
        return None

    body = ir.stmt_functor.ir_transform(
        kernel.body,
        validate_region,
        rewrite,
        ["tirx.BufferLoad", "tirx.BufferStore", "tirx.Call"],
    )
    return ir.PrimFunc(
        [*kernel.params, mapping.data], body, kernel.ret_type, buffers, kernel.attrs
    )


def make_kernel(module, factory, arguments, states, rows, capacity):
    from importlib import import_module

    kernel = getattr(import_module(module), factory)(*arguments)
    return map_state(kernel, states, rows=rows, capacity=capacity)
