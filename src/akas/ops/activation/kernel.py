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
    parts = p.get("partitions", 1)
    if (
        type(parts) is not int
        or parts < 1
        or parts > 1
        and (flat or packed or not expert_rows)
    ):
        raise ValueError("partitioned SwiGLU requires separate expert-row products")
    input_shape = (
        (r, top, parts, c)
        if parts > 1
        else (r, top, c)
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
        if parts > 1:
            gate = 0
            up = 0
            for part in T.serial(parts):
                gate += T.cast(g[row // top, row % top, part, column], "float32")
                up += T.cast(u[row // top, row % top, part, column], "float32")
        elif expert_rows:
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


def sigmoid_bias(spec, schedule):
    """Apply a shared scalar bias and sigmoid to projection results."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows = p["r"]

    @kernel
    def compute(x: Tensor((rows,), "float32"), bias: Tensor((1,), "float32")) -> Tensor(
        (rows,), "float32"
    ):
        out = output((rows,), "float32")
        with T.Kernel(T.ceildiv(rows, 128), threads=128) as block:
            for lane in T.Parallel(128):
                row = block * 128 + lane
                if row < rows:
                    out[row] = 1 / (1 + T.exp(-(x[row] + bias[0])))
        return out

    return compute


def geglu(spec, schedule):
    """Exact GELU(gate) times up, with FP32 intermediate arithmetic."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    n = p["r"] * p["c"]

    @kernel
    def compute(x: Tensor((n,), "float32"), y: Tensor((n,), "float32")) -> Tensor(
        (n,), "float32"
    ):
        out = output((n,), "float32")
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    out[i] = 0.5 * x[i] * (1 + T.erf(x[i] * 0.7071067811865476)) * y[i]
        return out

    return compute


def head_gate(spec, schedule):
    """Broadcast one sigmoid gate per query head."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, d = (p[k] for k in ("r", "h", "d"))
    n = r * h * d

    @kernel
    def compute(
        x: Tensor((n,), "float32"), gate: Tensor((r * h,), "float32")
    ) -> Tensor((n,), "float32"):
        out = output((n,), "float32")
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    out[i] = x[i] / (1 + T.exp(-gate[i // d]))
        return out

    return compute


def head_gate_half(spec, schedule):
    """Round the activation once at the following projection's MMA precision."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, d = (p[k] for k in ("r", "h", "d"))
    n = r * h * d

    @kernel
    def compute(
        x: Tensor((n,), "float32"), gate: Tensor((r * h,), "float32")
    ) -> Tensor((n,), "float16"):
        out = output((n,), "float16")
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    out[i] = x[i] / (1 + T.exp(-gate[i // d]))
        return out

    return compute


def gelu_erf():
    import tilelang.language as T

    @T.macro
    def activate(value):
        x = value * T.float32(0.7071067811865476)
        t = 1 / (1 + T.float32(0.3275911) * T.abs(x))
        poly = (
            ((T.float32(1.061405429) * t - 1.453152027) * t + 1.421413741) * t
            - 0.284496736
        ) * t + 0.254829592
        erf = 1 - poly * t * T.exp(-x * x)
        return 0.5 * value * (1 + T.if_then_else(x < 0, -erf, erf))

    return activate


def column_activation(spec, schedule):
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, c = (p[key] for key in ("r", "c"))
    gelu = spec.kind != "column_bias"
    activate = gelu_erf()

    if spec.kind == "gelu_bias" and p.get("output_dtype") == "float16":
        from akas.quantization._shared.rounding import round_half

        rounded = round_half()

        @kernel
        def compute(
            x: Tensor((r, c), "float32"), bias: Tensor((c,), "float32")
        ) -> Tensor((r, c), "float16"):
            out = output((r, c), "float16")
            with T.Kernel(r, T.ceildiv(c, 128), threads=128) as (row, tile):
                for lane in T.Parallel(128):
                    col = tile * 128 + lane
                    if col < c:
                        out[row, col] = rounded(activate(x[row, col] + bias[col]))
            return out

        return compute

    if spec.kind == "softplus_bias":

        @kernel
        def compute(
            x: Tensor((r, c), "float32"), bias: Tensor((c,), "float32")
        ) -> Tensor((r, c), "float32"):
            out = output((r, c), "float32")
            with T.Kernel(r, T.ceildiv(c, 128), threads=128) as (row, tile):
                for lane in T.Parallel(128):
                    col = tile * 128 + lane
                    if col < c:
                        value = x[row, col] + bias[col]
                        out[row, col] = T.log(1 + T.exp(-T.abs(value))) + T.max(
                            value, 0
                        )
            return out

        return compute

    if spec.kind == "silu":

        @kernel
        def compute(x: Tensor((r, c), "float32")) -> Tensor((r, c), "float32"):
            out = output((r, c), "float32")
            with T.Kernel(r, T.ceildiv(c, 128), threads=128) as (row, tile):
                for lane in T.Parallel(128):
                    col = tile * 128 + lane
                    if col < c:
                        out[row, col] = x[row, col] / (1 + T.exp(-x[row, col]))
            return out

        return compute

    if spec.kind == "gelu":

        @kernel
        def compute(x: Tensor((r, c), "float32")) -> Tensor((r, c), "float32"):
            out = output((r, c), "float32")
            with T.Kernel(r, T.ceildiv(c, 128), threads=128) as (row, tile):
                for lane in T.Parallel(128):
                    col = tile * 128 + lane
                    if col < c:
                        out[row, col] = activate(x[row, col])
            return out

        return compute

    @kernel
    def compute(x: Tensor((r, c), "float32"), bias: Tensor((c,), "float32")) -> Tensor(
        (r, c), "float32"
    ):
        out = output((r, c), "float32")
        with T.Kernel(r, T.ceildiv(c, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                col = tile * 128 + lane
                if col < c:
                    value = x[row, col] + bias[col]
                    out[row, col] = activate(value) if gelu else value
        return out

    return compute
