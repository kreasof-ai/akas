"""Shared TileLang computations for activation."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def swiglu(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.activation.kernel import _swiglu as swiglu_kernel

    p = {**spec.parameters, **schedule}
    parameters = dict(p)
    if parameters.get("packed"):
        parameters["r"] = p.get("slots", p.get("r", 1)) * p.get("chunk", 1)
    return swiglu_kernel(parameters)


def swiglu_experts(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.activation.kernel import _swiglu as swiglu_kernel

    p = {**spec.parameters, **schedule}
    parameters = dict(p)
    if parameters.get("packed"):
        parameters["r"] = p.get("slots", p.get("r", 1)) * p.get("chunk", 1)
    return swiglu_kernel(parameters)


def _swiglu(p):
    import tilelang.language as T

    r, c = (p["r"], p["c"])
    flat, packed = (p.get("flat", False), p.get("packed", False))
    expert_rows = p.get("expert_rows", False)
    top = p.get("top", 1)
    rounded = p.get("round_inputs", False)
    input_dtype = p.get("input_dtype", "float32")
    output_dtype = p.get("output_dtype", input_dtype)
    round_dtype = p.get("round_dtype", output_dtype)
    input_shape = (
        (r, top, c)
        if expert_rows
        else (r * c,)
        if flat
        else (r, (2 if packed else 1) * c)
    )
    output_shape = (r, top, c) if expert_rows else (r * c,) if flat else (r, c)
    if flat and packed:
        raise ValueError("packed SwiGLU uses a two-dimensional projection")

    @T.macro
    def element(g, u, out, row, column):
        gate = T.alloc_var("float32")
        up = T.alloc_var("float32")
        if expert_rows:
            gate = T.cast(g[row // top, row % top, column], "float32")
            up = T.cast(u[row // top, row % top, column], "float32")
        elif flat:
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
        if expert_rows:
            out[row // top, row % top, column] = gate / (1 + T.exp(-gate)) * up
        elif flat:
            out[row * c + column] = gate / (1 + T.exp(-gate)) * up
        else:
            out[row, column] = gate / (1 + T.exp(-gate)) * up

    @T.macro
    def separate(g, u, out):
        if expert_rows:
            with T.Kernel(r, top, T.ceildiv(c, 256), threads=256) as (row, rank, tile):
                for lane in T.Parallel(256):
                    column = tile * 256 + lane
                    if column < c:
                        element(g, u, out, row * top + rank, column)
        elif flat:
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

    names = p.get("input_names", ("g", "u"))
    if not packed:

        @kernel(names={"g": "x" if packed else names[0], "u": names[1]})
        def compute(
            g: Tensor(input_shape, input_dtype), u: Tensor(input_shape, input_dtype)
        ) -> Tensor(output_shape, output_dtype):
            out = output(output_shape, output_dtype)
            if expert_rows:
                with T.Kernel(r, top, T.ceildiv(c, 256), threads=256) as (
                    row,
                    rank,
                    tile,
                ):
                    for lane in T.Parallel(256):
                        column = tile * 256 + lane
                        if column < c:
                            element(g, u, out, row * top + rank, column)
            elif flat:
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
            return out

        return compute
    if packed:

        @kernel(names={"x": "x" if packed else names[0]})
        def compute(x: Tensor(input_shape, input_dtype)) -> Tensor(
            output_shape, output_dtype
        ):
            out = output(output_shape, output_dtype)
            separate(x, x, out)
            return out

        return compute
