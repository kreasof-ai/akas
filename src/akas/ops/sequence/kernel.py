"""Shared TileLang computations for sequence."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def mapped_controls(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))

    @kernel
    def compute(
        position: Tensor((slots,), "int32"), lengths: Tensor((slots,), "int32")
    ) -> tuple[Tensor((rows,), "int32"), Tensor((rows,), "int32")]:
        flat_position = output((rows,), "int32")
        active = output((rows,), "int32")
        with T.Kernel(T.ceildiv(rows, 256), threads=256) as block:
            for i in T.Parallel(256):
                row = block * 256 + i
                if row < rows:
                    flat_position[row] = position[row // chunk] + row % chunk
                    active[row] = T.cast(row % chunk < lengths[row // chunk], "int32")
        return (flat_position, active)

    return compute


def controls_chunk_controls(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk

    @kernel
    def compute(
        position: Tensor((slots,), "int32"), lengths: Tensor((slots,), "int32")
    ) -> tuple[Tensor((rows,), "int32"), Tensor((rows,), "int32")]:
        flat_position = output((rows,), "int32")
        active = output((rows,), "int32")
        with T.Kernel(T.ceildiv(rows, 256), threads=256) as tile:
            for j in T.Parallel(256):
                row = tile * 256 + j
                if row < rows:
                    flat_position[row] = position[row // chunk] + row % chunk
                    active[row] = T.cast(row % chunk < lengths[row // chunk], "int32")
        return (flat_position, active)

    return compute


def baseline_advance(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)

    @kernel
    def compute(pos: Tensor((2,), "int32")) -> Tensor((2,), "int32"):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(1):
                pos[0] += pos[1]
        return pos

    return compute


def chunk_advance(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    slots, chunk = (p["slots"], p["chunk"])
    slots, chunk = (p["slots"], p["chunk"])
    all_tokens = p.get("all_tokens", False)

    @kernel
    def compute(
        position: Tensor((slots,), "int32"), lengths: Tensor((slots,), "int32")
    ) -> tuple[Tensor((slots,), "int32"), Tensor((slots,), "int32")]:
        active = output((slots,), "int32", position=2)
        with T.Kernel(1, threads=32):
            for slot in T.Parallel(slots):
                active[slot] = T.cast(lengths[slot] > 0, "int32")
                position[slot] += T.max(lengths[slot] - (0 if all_tokens else 1), 0)
        return (position, active)

    return compute


def flat_prefill_tail(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c, t = (p["r"], p["c"], p["t"])

    @kernel
    def compute(
        x: Tensor((r * c,), "float32"), pos: Tensor((2,), "int32")
    ) -> tuple[Tensor((t * c,), "float32"), Tensor((2,), "int32")]:
        out = output((t * c,), "float32", position=1)
        tail_pos = output((2,), "int32", position=3)
        with T.Kernel(T.ceildiv(t * c, 256), threads=256) as bx:
            for lane in T.Parallel(256):
                i = bx * 256 + lane
                if i < t * c:
                    out[i] = T.if_then_else(
                        i // c < T.min(pos[1], t),
                        x[(T.max(pos[1] - t, 0) + i // c) * c + i % c],
                        0,
                    )
                if (bx == 0) & (lane == 0):
                    tail_pos[0] = pos[0] + T.max(pos[1] - t, 0)
                    tail_pos[1] = T.min(pos[1], t)
        return (out, tail_pos)

    return compute


def baseline_last(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)

    @kernel
    def compute(x: Tensor((r * c,), "float32"), pos: Tensor((2,), "int32")) -> Tensor(
        (c,), "float32"
    ):
        out = output((c,), "float32", position=1)
        with T.Kernel(T.ceildiv(c, 256), threads=256) as block:
            for lane in T.Parallel(256):
                col = block * 256 + lane
                if col < c:
                    out[col] = x[(pos[1] - 1) * c + col]
        return out

    return compute


def mapped_gather_hidden(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    c = p["c"]

    @kernel
    def compute(
        saved: Tensor((pool, c), "bfloat16"),
        mapping: Tensor((slots,), "int32"),
        mode: Tensor((1,), "int32"),
    ) -> Tensor((rows, c), "bfloat16"):
        out = output((rows, c), "bfloat16")
        with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
            if (mode[0] != 0) & (mapping[batch] >= 0):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        out[batch * chunk, col] = saved[mapping[batch], col]
        return out

    return compute


def mapped_last_rows(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    c = p["c"]

    @kernel
    def compute(
        x: Tensor((rows, c), "bfloat16"), lengths: Tensor((slots,), "int32")
    ) -> Tensor((slots, c), "bfloat16"):
        out = output((slots, c), "bfloat16")
        with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                if col < c:
                    out[batch, col] = T.if_then_else(
                        lengths[batch] > 0,
                        x[batch * chunk + T.max(lengths[batch] - 1, 0), col],
                        0,
                    )
        return out

    return compute


def mapped_store_hidden(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    c = p["c"]

    @kernel
    def compute(
        x: Tensor((slots, c), "bfloat16"),
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((pool, c), "bfloat16"):
        saved = output((pool, c), "bfloat16")
        with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
            if (mapping[batch] >= 0) & (lengths[batch] > 0):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        saved[mapping[batch], col] = x[batch, col]
        return saved

    return compute


def chunk_last_rows(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    slots, chunk = (p["slots"], p["chunk"])
    width = p["width"]

    @kernel
    def compute(
        x: Tensor((rows, width), "bfloat16"), lengths: Tensor((slots,), "int32")
    ) -> Tensor((slots, width), "bfloat16"):
        out = output((slots, width), "bfloat16")
        with T.Kernel(slots, T.ceildiv(width, 256), threads=256) as (slot, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                if col < width:
                    out[slot, col] = T.if_then_else(
                        lengths[slot] > 0,
                        x[slot * chunk + T.max(lengths[slot] - 1, 0), col],
                        0,
                    )
        return out

    return compute


def advance(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p["r"]
    chunk = p.get("chunk", 1)

    @kernel
    def compute(
        positions: Tensor((r,), "int32"), enabled: Tensor((r,), "int32")
    ) -> Tensor((r,), "int32"):
        with T.Kernel(T.ceildiv(r, 64), threads=64) as block:
            tx = T.get_thread_binding()
            row = block * 64 + tx
            if row < r:
                positions[row] += (
                    enabled[row]
                    if chunk > 1
                    else T.if_then_else(enabled[row] != 0, 1, 0)
                )
        return positions

    return compute


def chunk_controls(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p["r"]
    chunk = p.get("chunk", 1)

    @kernel
    def compute(
        positions: Tensor((r,), "int32"), lengths: Tensor((r,), "int32")
    ) -> tuple[Tensor((r * chunk,), "int32"), Tensor((r * chunk,), "int32")]:
        row_positions = output((r * chunk,), "int32")
        enabled = output((r * chunk,), "int32")
        with T.Kernel(T.ceildiv(r * chunk, 64), threads=64) as block:
            tx = T.get_thread_binding()
            row = block * 64 + tx
            if row < r * chunk:
                row_positions[row] = positions[row // chunk] + T.min(
                    row % chunk, T.max(lengths[row // chunk] - 1, 0)
                )
                enabled[row] = T.if_then_else(row % chunk < lengths[row // chunk], 1, 0)
        return (row_positions, enabled)

    return compute


def chunk_last(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p["r"]
    chunk = p.get("chunk", 1)
    c = p["c"]

    @kernel
    def compute(
        x: Tensor((r * chunk * c,), "float32"), lengths: Tensor((r,), "int32")
    ) -> Tensor((r * c,), "float32"):
        out = output((r * c,), "float32", position=1)
        with T.Kernel(T.ceildiv(c, 64), r, threads=64) as (block, slot):
            col = block * 64 + T.get_thread_binding()
            if col < c:
                out[slot * c + col] = x[
                    (slot * chunk + T.max(lengths[slot] - 1, 0)) * c + col
                ]
        return out

    return compute


def masked_copy(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p["r"]
    chunk = p.get("chunk", 1)
    c = p["c"]

    @kernel
    def compute(
        x: Tensor((r * c,), "float32"), enabled: Tensor((r,), "int32")
    ) -> Tensor((r * c,), "float32"):
        out = output((r * c,), "float32", position=1)
        with T.Kernel(T.ceildiv(c, 128), r, threads=128) as (block, row):
            col = block * 128 + T.get_thread_binding()
            if col < c and enabled[row] != 0:
                out[row * c + col] = x[row * c + col]
        return out

    return compute


def next_tokens(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p["r"]
    chunk = p.get("chunk", 1)

    @kernel
    def compute(sampled: Tensor((r,), "int32")) -> Tensor((r,), "int32"):
        tokens = output((r,), "int32")
        with T.Kernel(T.ceildiv(r, 64), threads=64) as block:
            tx = T.get_thread_binding()
            row = block * 64 + tx
            if row < r:
                tokens[row] = sampled[row]
        return tokens

    return compute


def chunk_tail(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    rows, chunk, tail = (p["r"], p["chunk"], p["tail"])
    if not 0 < tail <= chunk:
        raise ValueError("suffix length must be within the input chunk")
    width = p["c"]

    @kernel
    def compute(
        x: Tensor((rows * chunk * width,), "float32"), lengths: Tensor((rows,), "int32")
    ) -> Tensor((rows * tail * width,), "float32"):
        out = output((rows * tail * width,), "float32")
        with T.Kernel(T.ceildiv(tail * width, 64), rows, threads=64) as (block, row):
            index = block * 64 + T.get_thread_binding()
            if index < tail * width:
                token = index // width
                if token < T.min(lengths[row], tail):
                    out[row * tail * width + index] = x[
                        (row * chunk + T.max(lengths[row] - tail, 0) + token) * width
                        + index % width
                    ]
                else:
                    out[row * tail * width + index] = 0
        return out

    return compute


def tail_controls(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    rows, chunk, tail = (p["r"], p["chunk"], p["tail"])
    if not 0 < tail <= chunk:
        raise ValueError("suffix length must be within the input chunk")

    @kernel
    def compute(
        positions: Tensor((rows * chunk,), "int32"), lengths: Tensor((rows,), "int32")
    ) -> tuple[
        Tensor((rows * tail,), "int32"),
        Tensor((rows * tail,), "int32"),
        Tensor((rows,), "int32"),
    ]:
        outpositions = output((rows * tail,), "int32")
        outactive = output((rows * tail,), "int32")
        outlengths = output((rows,), "int32")
        with T.Kernel(T.ceildiv(tail, 64), rows, threads=64) as (block, row):
            token = block * 64 + T.get_thread_binding()
            if token < tail:
                if token < T.min(lengths[row], tail):
                    outpositions[row * tail + token] = positions[
                        row * chunk + T.max(lengths[row] - tail, 0) + token
                    ]
                    outactive[row * tail + token] = 1
                else:
                    outpositions[row * tail + token] = 0
                    outactive[row * tail + token] = 0
            if token == 0:
                outlengths[row] = T.min(lengths[row], tail)
        return (outpositions, outactive, outlengths)

    return compute


def prefill_tail(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    c, t = (p["c"], p["t"])

    @kernel
    def compute(
        x: Tensor((r * c,), "float32"), control: Tensor((2,), "int32")
    ) -> tuple[Tensor((t * c,), "float32"), Tensor((2,), "int32")]:
        out = output((t * c,), "float32", position=1)
        tail_control = output((2,), "int32", position=3)
        with T.Kernel(T.ceildiv(t * c, 256), threads=256) as block:
            tx = T.get_thread_binding()
            index = block * 256 + tx
            start = T.max(control[1] - t, 0)
            count = T.min(control[1], t)
            if index < t * c:
                out[index] = T.if_then_else(index // c < count, x[start * c + index], 0)
            if index == 0:
                tail_control[0] = control[0] + start
                tail_control[1] = count
        return (out, tail_control)

    return compute
