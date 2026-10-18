"""Shared TileLang computations for gdn."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def mapped_conv(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    c, projection = (p["channels"], p["projection"])

    @kernel
    def compute(
        x: Tensor((rows, projection), "float32"),
        w: Tensor((c, 1, 4), "bfloat16"),
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((pool, c, 3), "float32"),
    ) -> tuple[Tensor((pool, c, 3), "float32"), Tensor((rows, c), "float32")]:
        out = output((rows, c), "float32")
        with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
            request = mapping[batch]
            if request >= 0:
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        h0 = T.alloc_var("float32")
                        h1 = T.alloc_var("float32")
                        h2 = T.alloc_var("float32")
                        h0 = state[request, col, 0]
                        h1 = state[request, col, 1]
                        h2 = state[request, col, 2]
                        for t in T.serial(chunk):
                            if t < lengths[batch]:
                                current = T.cast(
                                    T.cast(x[batch * chunk + t, col], "bfloat16"),
                                    "float32",
                                )
                                value = h0 * T.cast(
                                    w[col, 0, 0], "float32"
                                ) + h1 * T.cast(w[col, 0, 1], "float32")
                                value += h2 * T.cast(
                                    w[col, 0, 2], "float32"
                                ) + current * T.cast(w[col, 0, 3], "float32")
                                out[batch * chunk + t, col] = T.cast(
                                    value / (1 + T.exp(-value)), "bfloat16"
                                )
                                h0 = h1
                                h1 = h2
                                h2 = current
                            else:
                                out[batch * chunk + t, col] = 0
                        if p.get("commit", True):
                            state[request, col, 0] = h0
                            state[request, col, 1] = h1
                            state[request, col, 2] = h2
        return (state, out)

    return compute


def mapped_conv_f16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    c, projection = (p["channels"], p["projection"])
    dtype = p.get("dtype", "float16")

    @kernel
    def compute(
        x: Tensor((rows, projection), "float32"),
        w: Tensor((c, 1, 4), dtype),
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((pool, c, 3), "float32"),
    ) -> tuple[Tensor((pool, c, 3), "float32"), Tensor((rows, c), "float32")]:
        out = output((rows, c), "float32")
        with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
            request = mapping[batch]
            if request >= 0:
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        h0 = T.alloc_var("float32")
                        h1 = T.alloc_var("float32")
                        h2 = T.alloc_var("float32")
                        h0 = state[request, col, 0]
                        h1 = state[request, col, 1]
                        h2 = state[request, col, 2]
                        for t in T.serial(chunk):
                            if t < lengths[batch]:
                                current = T.cast(
                                    T.cast(x[batch * chunk + t, col], dtype),
                                    "float32",
                                )
                                value = h0 * T.cast(
                                    w[col, 0, 0], "float32"
                                ) + h1 * T.cast(w[col, 0, 1], "float32")
                                value += h2 * T.cast(
                                    w[col, 0, 2], "float32"
                                ) + current * T.cast(w[col, 0, 3], "float32")
                                out[batch * chunk + t, col] = T.cast(
                                    value / (1 + T.exp(-value)), dtype
                                )
                                h0 = h1
                                h1 = h2
                                h2 = current
                            else:
                                out[batch * chunk + t, col] = 0
                        if p.get("commit", True):
                            state[request, col, 0] = h0
                            state[request, col, 1] = h1
                            state[request, col, 2] = h2
        return (state, out)

    return compute


def decode_conv(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, channels = (p["r"], 8192)

    @kernel
    def compute(
        x: Tensor((r, channels), "float32"),
        w: Tensor((channels, 1, 4), "bfloat16"),
        active: Tensor((r,), "int32"),
        state: Tensor((r, channels, 3), "float32"),
    ) -> tuple[Tensor((r, channels, 3), "float32"), Tensor((r, channels), "float32")]:
        out = output((r, channels), "float32")
        with T.Kernel(r, channels // 256, threads=256) as (b, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                current = T.cast(T.cast(x[b, col], "bfloat16"), "float32")
                value = T.alloc_var("float32")
                value = current * T.cast(w[col, 0, 3], "float32")
                for t in T.unroll(3):
                    value += state[b, col, t] * T.cast(w[col, 0, t], "float32")
                if active[b] != 0:
                    state[b, col, 0] = state[b, col, 1]
                    state[b, col, 1] = state[b, col, 2]
                    state[b, col, 2] = current
                    out[b, col] = T.cast(value / (1 + T.exp(-value)), "bfloat16")
                else:
                    out[b, col] = 0
        return (state, out)

    return compute


def prefill_conv(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    commit = p.get("commit", True)
    channels = 8192

    @kernel
    def compute(
        x: Tensor((rows, channels), "float32"),
        w: Tensor((channels, 1, 4), "bfloat16"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((slots, channels, 3), "float32"),
    ) -> tuple[
        Tensor((slots, channels, 3), "float32"), Tensor((rows, channels), "float32")
    ]:
        out = output((rows, channels), "float32")
        with T.Kernel(slots, channels // 256, threads=256) as (slot, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                h0 = T.alloc_var("float32")
                h1 = T.alloc_var("float32")
                h2 = T.alloc_var("float32")
                h0 = state[slot, col, 0]
                h1 = state[slot, col, 1]
                h2 = state[slot, col, 2]
                for t in T.serial(chunk):
                    if t < lengths[slot]:
                        current = T.cast(
                            T.cast(x[slot * chunk + t, col], "bfloat16"), "float32"
                        )
                        value = current * T.cast(w[col, 0, 3], "float32")
                        value += h0 * T.cast(w[col, 0, 0], "float32")
                        value += h1 * T.cast(w[col, 0, 1], "float32")
                        value += h2 * T.cast(w[col, 0, 2], "float32")
                        out[slot * chunk + t, col] = T.cast(
                            value / (1 + T.exp(-value)), "bfloat16"
                        )
                        h0 = h1
                        h1 = h2
                        h2 = current
                    else:
                        out[slot * chunk + t, col] = 0
                if commit:
                    state[slot, col, 0] = h0
                    state[slot, col, 1] = h1
                    state[slot, col, 2] = h2
        return (state, out)

    return compute


def journal_conv(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    snapshots = chunk if p.get("full_checkpoints", False) else chunk - 1
    commit = p.get("commit", True)
    channels = 8192

    @kernel
    def compute(
        x: Tensor((rows, channels), "float32"),
        w: Tensor((channels, 1, 4), "bfloat16"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((slots, channels, 3), "float32"),
    ) -> tuple[
        Tensor((slots, channels, 3), "float32"),
        Tensor((rows, channels), "float32"),
        Tensor((slots, snapshots, channels, 3), "float32"),
    ]:
        out = output((rows, channels), "float32")
        checkpoints = output((slots, snapshots, channels, 3), "float32")
        with T.Kernel(slots, channels // 256, threads=256) as (slot, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                h0 = T.alloc_var("float32")
                h1 = T.alloc_var("float32")
                h2 = T.alloc_var("float32")
                h0 = state[slot, col, 0]
                h1 = state[slot, col, 1]
                h2 = state[slot, col, 2]
                for t in T.serial(chunk):
                    if t < lengths[slot]:
                        current = T.cast(
                            T.cast(x[slot * chunk + t, col], "bfloat16"), "float32"
                        )
                        value = current * T.cast(w[col, 0, 3], "float32")
                        value += h0 * T.cast(w[col, 0, 0], "float32")
                        value += h1 * T.cast(w[col, 0, 1], "float32")
                        value += h2 * T.cast(w[col, 0, 2], "float32")
                        out[slot * chunk + t, col] = T.cast(
                            value / (1 + T.exp(-value)), "bfloat16"
                        )
                        h0 = h1
                        h1 = h2
                        h2 = current
                        if t < snapshots:
                            checkpoints[slot, t, col, 0] = h0
                            checkpoints[slot, t, col, 1] = h1
                            checkpoints[slot, t, col, 2] = h2
                    else:
                        out[slot * chunk + t, col] = 0
                if commit:
                    state[slot, col, 0] = h0
                    state[slot, col, 1] = h1
                    state[slot, col, 2] = h2
        return (state, out, checkpoints)

    return compute


def conv_commit(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window, pool = (p[n] for n in ("slots", "window", "pool"))
    heads, d, tile = (p.get("heads", 16), p.get("d", 128), p.get("tile", 32))
    channels, projection = (p["channels"], p["projection"])

    @kernel
    def compute(
        x: Tensor((slots * window, projection), "float32"),
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((pool, channels, 3), "float32"),
    ) -> Tensor((pool, channels, 3), "float32"):
        with T.Kernel(slots, T.ceildiv(channels, 256), threads=256) as (batch, tile):
            request = mapping[batch]
            if request >= 0:
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < channels:
                        h0 = T.alloc_var("float32")
                        h1 = T.alloc_var("float32")
                        h2 = T.alloc_var("float32")
                        h0 = state[request, col, 0]
                        h1 = state[request, col, 1]
                        h2 = state[request, col, 2]
                        for t in T.serial(window):
                            if t < lengths[batch]:
                                h0 = h1
                                h1 = h2
                                h2 = T.cast(
                                    T.cast(x[batch * window + t, col], "bfloat16"),
                                    "float32",
                                )
                        state[request, col, 0] = h0
                        state[request, col, 1] = h1
                        state[request, col, 2] = h2
        return state

    return compute


def conv_commit_f16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window, pool = (p[n] for n in ("slots", "window", "pool"))
    heads, d, tile = (p.get("heads", 16), p.get("d", 128), p.get("tile", 32))
    channels, projection = (p["channels"], p["projection"])
    dtype = p.get("dtype", "float16")

    @kernel
    def compute(
        x: Tensor((slots * window, projection), "float32"),
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((pool, channels, 3), "float32"),
    ) -> Tensor((pool, channels, 3), "float32"):
        with T.Kernel(slots, T.ceildiv(channels, 256), threads=256) as (batch, tile):
            request = mapping[batch]
            if request >= 0:
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < channels:
                        h0 = T.alloc_var("float32")
                        h1 = T.alloc_var("float32")
                        h2 = T.alloc_var("float32")
                        h0 = state[request, col, 0]
                        h1 = state[request, col, 1]
                        h2 = state[request, col, 2]
                        for t in T.serial(window):
                            if t < lengths[batch]:
                                h0 = h1
                                h1 = h2
                                h2 = T.cast(
                                    T.cast(x[batch * window + t, col], dtype),
                                    "float32",
                                )
                        state[request, col, 0] = h0
                        state[request, col, 1] = h1
                        state[request, col, 2] = h2
        return state

    return compute


def defer_accept(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window, pool = (p[n] for n in ("slots", "window", "pool"))
    heads, d, tile = (p.get("heads", 16), p.get("d", 128), p.get("tile", 32))

    @kernel
    def compute(
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
        counts: Tensor((pool,), "int32"),
        banks: Tensor((pool,), "int32"),
    ) -> tuple[Tensor((pool,), "int32"), Tensor((pool,), "int32")]:
        with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
            for j in T.Parallel(128):
                row = block * 128 + j
                if row < slots:
                    request = mapping[row]
                    if request >= 0:
                        counts[request] = T.max(lengths[row] - 1, 0)
                        banks[request] = 1 - banks[request]
        return (counts, banks)

    return compute


def defer_clear(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window, pool = (p[n] for n in ("slots", "window", "pool"))
    heads, d, tile = (p.get("heads", 16), p.get("d", 128), p.get("tile", 32))

    @kernel
    def compute(
        mapping: Tensor((slots,), "int32"), counts: Tensor((pool,), "int32")
    ) -> Tensor((pool,), "int32"):
        with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
            for j in T.Parallel(128):
                row = block * 128 + j
                if row < slots:
                    request = mapping[row]
                    if request >= 0:
                        counts[request] = 0
        return counts

    return compute


def gdn_deferred(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window, pool = (p[n] for n in ("slots", "window", "pool"))
    heads, d, tile = (
        p.get("heads", 16),
        p.get("d", 128),
        p.get("webgpu_tile", p.get("tile", 32)),
    )
    rows = slots * window

    @kernel
    def compute(
        q: Tensor((rows, heads, d), "float32"),
        k: Tensor((rows, heads, d), "float32"),
        v: Tensor((rows, heads, d), "float32"),
        g: Tensor((rows, heads), "float32"),
        beta: Tensor((rows, heads), "float32"),
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((pool, heads, d, d), "float32"),
        saved_k: Tensor((2, pool, window, heads, d), "float32"),
        saved_v: Tensor((2, pool, window, heads, d), "float32"),
        saved_g: Tensor((2, pool, window, heads), "float32"),
        saved_b: Tensor((2, pool, window, heads), "float32"),
        counts: Tensor((pool,), "int32"),
        banks: Tensor((pool,), "int32"),
    ) -> tuple[
        Tensor((pool, heads, d, d), "float32"),
        Tensor((rows, heads, d), "float32"),
        Tensor((2, pool, window, heads, d), "float32"),
        Tensor((2, pool, window, heads, d), "float32"),
        Tensor((2, pool, window, heads), "float32"),
        Tensor((2, pool, window, heads), "float32"),
    ]:
        out = output((rows, heads, d), "float32", position=8)
        with T.Kernel(slots, heads, d // tile, threads=128) as (batch, head, part):
            request = mapping[batch]
            if request >= 0:
                bank = banks[request]
                matrix = T.alloc_fragment((tile, d), "float32")
                prediction = T.alloc_fragment((tile, d), "float32")
                inner = T.alloc_fragment((tile,), "float32")
                correction = T.alloc_fragment((tile,), "float32")
                result = T.alloc_fragment((tile,), "float32")
                qq = T.alloc_shared((d,), "float32")
                kk = T.alloc_shared((d,), "float32")
                T.copy(
                    state[request, head, part * tile : part * tile + tile, :], matrix
                )
                for t in T.serial(window):
                    if t < counts[request]:
                        for j in T.Parallel(d):
                            kk[j] = saved_k[bank, request, t, head, j]
                        for i, j in T.Parallel(tile, d):
                            matrix[i, j] *= T.exp(saved_g[bank, request, t, head])
                            matrix[i, j] += (
                                kk[j]
                                * saved_b[bank, request, t, head]
                                * saved_v[bank, request, t, head, part * tile + i]
                            )
                for t in T.serial(window):
                    row = batch * window + t
                    if t < lengths[batch]:
                        for j in T.Parallel(d):
                            qq[j] = q[row, head, j]
                            kk[j] = k[row, head, j]
                        for i, j in T.Parallel(tile, d):
                            matrix[i, j] *= T.exp(g[row, head])
                            prediction[i, j] = matrix[i, j] * kk[j]
                        T.reduce_sum(prediction, inner, dim=1)
                        for i in T.Parallel(tile):
                            correction[i] = v[row, head, part * tile + i] - inner[i]
                            if t > 0:
                                saved_v[
                                    1 - bank, request, t - 1, head, part * tile + i
                                ] = correction[i]
                        for i, j in T.Parallel(tile, d):
                            matrix[i, j] += kk[j] * beta[row, head] * correction[i]
                            prediction[i, j] = matrix[i, j] * qq[j]
                        T.reduce_sum(prediction, result, dim=1)
                        for i in T.Parallel(tile):
                            out[row, head, part * tile + i] = result[i]
                        if t == 0:
                            T.copy(
                                matrix,
                                state[
                                    request, head, part * tile : part * tile + tile, :
                                ],
                            )
                        if (part == 0) & (t > 0):
                            for j in T.Parallel(d):
                                saved_k[1 - bank, request, t - 1, head, j] = kk[j]
                            for j in T.Parallel(1):
                                saved_g[1 - bank, request, t - 1, head] = g[row, head]
                                saved_b[1 - bank, request, t - 1, head] = beta[
                                    row, head
                                ]
                    else:
                        for i in T.Parallel(tile):
                            out[row, head, part * tile + i] = 0
        return (state, out, saved_k, saved_v, saved_g, saved_b)

    return compute


def gdn_materialize(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window, pool = (p[n] for n in ("slots", "window", "pool"))
    heads, d, tile = (
        p.get("heads", 16),
        p.get("d", 128),
        p.get("webgpu_tile", p.get("tile", 32)),
    )

    @kernel
    def compute(
        mapping: Tensor((slots,), "int32"),
        state: Tensor((pool, heads, d, d), "float32"),
        saved_k: Tensor((2, pool, window, heads, d), "float32"),
        saved_v: Tensor((2, pool, window, heads, d), "float32"),
        saved_g: Tensor((2, pool, window, heads), "float32"),
        saved_b: Tensor((2, pool, window, heads), "float32"),
        counts: Tensor((pool,), "int32"),
        banks: Tensor((pool,), "int32"),
    ) -> Tensor((pool, heads, d, d), "float32"):
        with T.Kernel(slots, heads, d // tile, threads=128) as (batch, head, part):
            request = mapping[batch]
            if request >= 0:
                if counts[request] > 0:
                    bank = banks[request]
                    matrix = T.alloc_fragment((tile, d), "float32")
                    kk = T.alloc_shared((d,), "float32")
                    T.copy(
                        state[request, head, part * tile : part * tile + tile, :],
                        matrix,
                    )
                    for t in T.serial(window):
                        if t < counts[request]:
                            for j in T.Parallel(d):
                                kk[j] = saved_k[bank, request, t, head, j]
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] *= T.exp(saved_g[bank, request, t, head])
                                matrix[i, j] += (
                                    kk[j]
                                    * saved_b[bank, request, t, head]
                                    * saved_v[bank, request, t, head, part * tile + i]
                                )
                    T.copy(
                        matrix,
                        state[request, head, part * tile : part * tile + tile, :],
                    )
        return state

    return compute


def mapped_norm(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, d, channels, projection, eps = (
        p[n] for n in ("heads", "d", "channels", "projection", "eps")
    )

    @kernel
    def compute(
        x: Tensor((rows, heads, d), "float32"),
        proj: Tensor((rows, projection), "float32"),
        w: Tensor((d,), "float32"),
    ) -> Tensor((rows, heads * d), "bfloat16"):
        out = output((rows, heads * d), "bfloat16")
        with T.Kernel(rows, heads, threads=128) as (row, head):
            values = T.alloc_fragment((d,), "float32")
            squares = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(d):
                values[j] = T.cast(T.cast(x[row, head, j], "bfloat16"), "float32")
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(d):
                gate = T.cast(
                    T.cast(proj[row, channels + head * d + j], "bfloat16"), "float32"
                )
                out[row, head * d + j] = (
                    values[j]
                    * T.rsqrt(total[0] / d + eps)
                    * w[j]
                    * gate
                    / (1 + T.exp(-gate))
                )
        return out

    return compute


def mapped_norm_f16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, d, channels, projection, eps = (
        p[n] for n in ("heads", "d", "channels", "projection", "eps")
    )
    dtype = p.get("dtype", "float16")

    @kernel
    def compute(
        x: Tensor((rows, heads, d), "float32"),
        proj: Tensor((rows, projection), "float32"),
        w: Tensor((d,), "float32"),
    ) -> Tensor((rows, heads * d), dtype):
        out = output((rows, heads * d), dtype)
        with T.Kernel(rows, heads, threads=128) as (row, head):
            values = T.alloc_fragment((d,), "float32")
            squares = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(d):
                values[j] = T.cast(T.cast(x[row, head, j], dtype), "float32")
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(d):
                gate = T.cast(
                    T.cast(proj[row, channels + head * d + j], dtype), "float32"
                )
                out[row, head * d + j] = (
                    values[j]
                    * T.rsqrt(total[0] / d + eps)
                    * w[j]
                    * gate
                    / (1 + T.exp(-gate))
                )
        return out

    return compute


def decode_norm(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, heads, d, eps = (p["r"], 32, 128, p["eps"])

    @kernel
    def compute(
        x: Tensor((r, heads, d), "float32"),
        z: Tensor((r, 4096), "float32"),
        w: Tensor((d,), "float32"),
    ) -> Tensor((r, 4096), "bfloat16"):
        out = output((r, 4096), "bfloat16")
        with T.Kernel(r, heads, threads=128) as (slot, head):
            values = T.alloc_fragment((d,), "float32")
            squares = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(d):
                values[j] = T.cast(T.cast(x[slot, head, j], "bfloat16"), "float32")
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(d):
                gate = T.cast(T.cast(z[slot, head * d + j], "bfloat16"), "float32")
                out[slot, head * d + j] = (
                    values[j]
                    * T.rsqrt(total[0] / d + eps)
                    * w[j]
                    * gate
                    / (1 + T.exp(-gate))
                )
        return out

    return compute


def mapped_prepare(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, kh, d, channels, projection = (
        p[n] for n in ("heads", "key_heads", "d", "channels", "projection")
    )
    value_width = heads * d

    @kernel
    def compute(
        conv: Tensor((rows, channels), "float32"),
        proj: Tensor((rows, projection), "float32"),
        dt: Tensor((heads,), "bfloat16"),
        A: Tensor((heads,), "float32"),
    ) -> tuple[
        Tensor((rows, heads, d), "float32"),
        Tensor((rows, heads, d), "float32"),
        Tensor((rows, heads, d), "float32"),
        Tensor((rows, heads), "float32"),
        Tensor((rows, heads), "float32"),
    ]:
        q = output((rows, heads, d), "float32")
        k = output((rows, heads, d), "float32")
        v = output((rows, heads, d), "float32")
        g = output((rows, heads), "float32")
        beta = output((rows, heads), "float32")
        with T.Kernel(rows, heads, threads=128) as (row, head):
            qq = T.alloc_fragment((d,), "float32")
            kk = T.alloc_fragment((d,), "float32")
            qs = T.alloc_fragment((d,), "float32")
            ks = T.alloc_fragment((d,), "float32")
            qt = T.alloc_fragment((1,), "float32")
            kt = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(d):
                qq[j] = conv[row, head // (heads // kh) * d + j]
                kk[j] = conv[row, kh * d + head // (heads // kh) * d + j]
                qs[j] = qq[j] * qq[j]
                ks[j] = kk[j] * kk[j]
            T.reduce_sum(qs, qt, dim=0)
            T.reduce_sum(ks, kt, dim=0)
            for j in T.Parallel(d):
                q[row, head, j] = (
                    qq[j] * T.rsqrt(qt[0] + T.float32(1e-06)) * d ** (-0.5)
                )
                k[row, head, j] = kk[j] * T.rsqrt(kt[0] + T.float32(1e-06))
                v[row, head, j] = conv[row, 2 * kh * d + head * d + j]
            for j in T.Parallel(1):
                a = T.cast(
                    T.cast(proj[row, channels + value_width + head], "bfloat16"),
                    "float32",
                ) + T.cast(dt[head], "float32")
                g[row, head] = -T.exp(A[head]) * (
                    T.max(a, 0) + T.log(1 + T.exp(-T.abs(a)))
                )
                beta[row, head] = 1 / (
                    1
                    + T.exp(
                        -T.cast(
                            T.cast(
                                proj[row, channels + value_width + heads + head],
                                "bfloat16",
                            ),
                            "float32",
                        )
                    )
                )
        return (q, k, v, g, beta)

    return compute


def mapped_prepare_f16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, kh, d, channels, projection = (
        p[n] for n in ("heads", "key_heads", "d", "channels", "projection")
    )
    value_width = heads * d
    dtype = p.get("dtype", "float16")

    @kernel
    def compute(
        conv: Tensor((rows, channels), "float32"),
        proj: Tensor((rows, projection), "float32"),
        dt: Tensor((heads,), dtype),
        A: Tensor((heads,), "float32"),
    ) -> tuple[
        Tensor((rows, heads, d), "float32"),
        Tensor((rows, heads, d), "float32"),
        Tensor((rows, heads, d), "float32"),
        Tensor((rows, heads), "float32"),
        Tensor((rows, heads), "float32"),
    ]:
        q = output((rows, heads, d), "float32")
        k = output((rows, heads, d), "float32")
        v = output((rows, heads, d), "float32")
        g = output((rows, heads), "float32")
        beta = output((rows, heads), "float32")
        with T.Kernel(rows, heads, threads=128) as (row, head):
            qq = T.alloc_fragment((d,), "float32")
            kk = T.alloc_fragment((d,), "float32")
            qs = T.alloc_fragment((d,), "float32")
            ks = T.alloc_fragment((d,), "float32")
            qt = T.alloc_fragment((1,), "float32")
            kt = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(d):
                qq[j] = conv[row, head // (heads // kh) * d + j]
                kk[j] = conv[row, kh * d + head // (heads // kh) * d + j]
                qs[j] = qq[j] * qq[j]
                ks[j] = kk[j] * kk[j]
            T.reduce_sum(qs, qt, dim=0)
            T.reduce_sum(ks, kt, dim=0)
            for j in T.Parallel(d):
                q[row, head, j] = (
                    qq[j] * T.rsqrt(qt[0] + T.float32(1e-06)) * d ** (-0.5)
                )
                k[row, head, j] = kk[j] * T.rsqrt(kt[0] + T.float32(1e-06))
                v[row, head, j] = conv[row, 2 * kh * d + head * d + j]
            for j in T.Parallel(1):
                a = T.cast(
                    T.cast(proj[row, channels + value_width + head], dtype),
                    "float32",
                ) + T.cast(dt[head], "float32")
                g[row, head] = -T.exp(A[head]) * (
                    T.max(a, 0) + T.log(1 + T.exp(-T.abs(a)))
                )
                beta[row, head] = 1 / (
                    1
                    + T.exp(
                        -T.cast(
                            T.cast(
                                proj[row, channels + value_width + heads + head],
                                dtype,
                            ),
                            "float32",
                        )
                    )
                )
        return (q, k, v, g, beta)

    return compute


def decode_prepare(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, heads, d = (p["r"], 32, 128)

    @kernel
    def compute(
        qkv: Tensor((r, 8192), "float32"),
        a: Tensor((r, heads), "float32"),
        b: Tensor((r, heads), "float32"),
        dt: Tensor((heads,), "bfloat16"),
        A: Tensor((heads,), "float32"),
    ) -> tuple[
        Tensor((r, heads, d), "float32"),
        Tensor((r, heads, d), "float32"),
        Tensor((r, heads, d), "float32"),
        Tensor((r, heads), "float32"),
        Tensor((r, heads), "float32"),
    ]:
        q = output((r, heads, d), "float32")
        k = output((r, heads, d), "float32")
        v = output((r, heads, d), "float32")
        g = output((r, heads), "float32")
        beta = output((r, heads), "float32")
        with T.Kernel(r, heads, threads=128) as (slot, head):
            qs = T.alloc_fragment((d,), "float32")
            ks = T.alloc_fragment((d,), "float32")
            qq = T.alloc_fragment((d,), "float32")
            kk = T.alloc_fragment((d,), "float32")
            qtotal = T.alloc_fragment((1,), "float32")
            ktotal = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(d):
                qq[j] = qkv[slot, head // 2 * d + j]
                kk[j] = qkv[slot, 2048 + head // 2 * d + j]
                qs[j] = qq[j] * qq[j]
                ks[j] = kk[j] * kk[j]
            T.reduce_sum(qs, qtotal, dim=0)
            T.reduce_sum(ks, ktotal, dim=0)
            for j in T.Parallel(d):
                q[slot, head, j] = (
                    qq[j] * T.rsqrt(qtotal[0] + T.float32(1e-06)) * d ** (-0.5)
                )
                k[slot, head, j] = kk[j] * T.rsqrt(ktotal[0] + T.float32(1e-06))
                v[slot, head, j] = qkv[slot, 4096 + head * d + j]
            for j in T.Parallel(1):
                value = T.cast(T.cast(a[slot, head], "bfloat16"), "float32") + T.cast(
                    dt[head], "float32"
                )
                softplus = T.max(value, 0) + T.log(1 + T.exp(-T.abs(value)))
                g[slot, head] = -T.exp(A[head]) * softplus
                beta[slot, head] = 1 / (
                    1 + T.exp(-T.cast(T.cast(b[slot, head], "bfloat16"), "float32"))
                )
        return (q, k, v, g, beta)

    return compute


def save_scan(p):
    import tilelang.language as T

    s, c = (p["slots"], p["chunk"])
    r = s * c
    h, k, v = (32, 128, 128)
    state_count = s * h * v * k
    input_count = r * h * k
    small_count = r * h

    @kernel
    def compute(
        state: Tensor((s, h, v, k), "float32"),
        query: Tensor((r, h, k), "float32"),
        key: Tensor((r, h, k), "float32"),
        value: Tensor((r, h, v), "float32"),
        g: Tensor((r, h), "float32"),
        beta: Tensor((r, h), "float32"),
        saved_k: Tensor((r, h, k), "float32"),
        saved_v: Tensor((r, h, v), "float32"),
        saved_g: Tensor((r, h), "float32"),
    ) -> tuple[
        Tensor((s, h, v, k), "float32"),
        Tensor((r, h, k), "float32"),
        Tensor((r, h, k), "float32"),
        Tensor((r, h, v), "float32"),
        Tensor((r, h), "float32"),
        Tensor((r, h), "float32"),
    ]:
        initial = output((s, h, v, k), "float32", position=6)
        saved_q = output((r, h, k), "float32", position=7)
        saved_beta = output((r, h), "float32", position=11)
        with T.Kernel(
            T.ceildiv(max(state_count, input_count), 1024), threads=256
        ) as block:
            source = T.view(state, shape=(state_count,))
            destination = T.view(initial, shape=(state_count,))
            qq = T.view(query, shape=(input_count,))
            sq = T.view(saved_q, shape=(input_count,))
            kk = T.view(key, shape=(input_count,))
            sk = T.view(saved_k, shape=(input_count,))
            vv = T.view(value, shape=(input_count,))
            sv = T.view(saved_v, shape=(input_count,))
            gg = T.view(g, shape=(small_count,))
            sg = T.view(saved_g, shape=(small_count,))
            bb = T.view(beta, shape=(small_count,))
            sb = T.view(saved_beta, shape=(small_count,))
            for j in T.Parallel(1024):
                index = block * 1024 + j
                if index < state_count:
                    destination[index] = source[index]
                if index < input_count:
                    sq[index] = qq[index]
                    sk[index] = kk[index]
                    sv[index] = vv[index]
                if index < small_count:
                    sg[index] = gg[index]
                    sb[index] = bb[index]
        return (initial, saved_q, saved_k, saved_v, saved_g, saved_beta)

    return compute


def save_conv(p):
    import tilelang.language as T

    s, c = (p["slots"], p["chunk"])
    r = s * c
    channels = 8192
    count = r * channels
    state_count = s * channels * 3

    @kernel
    def compute(
        x: Tensor((r, channels), "float32"), state: Tensor((s, channels, 3), "float32")
    ) -> tuple[Tensor((r, channels), "bfloat16"), Tensor((s, channels, 3), "float32")]:
        saved_x = output((r, channels), "bfloat16")
        initial = output((s, channels, 3), "float32")
        with T.Kernel(T.ceildiv(max(count, state_count), 1024), threads=256) as block:
            source = T.view(state, shape=(state_count,))
            destination = T.view(initial, shape=(state_count,))
            for j in T.Parallel(1024):
                index = block * 1024 + j
                if index < count:
                    saved_x[index // channels, index % channels] = x[
                        index // channels, index % channels
                    ]
                if index < state_count:
                    destination[index] = source[index]
        return (saved_x, initial)

    return compute


def restore_scan(p):
    """Reset only partial slots; the unchanged scan then replays their inputs.

    Using the original scan retains its reduction layout and FP32 rounding.
    """
    import tilelang.language as T

    s, c = (p["slots"], p["chunk"])
    h, k, v = (32, 128, 128)
    count = h * k * v

    @kernel
    def compute(
        initial: Tensor((s, h, v, k), "float32"),
        accepted: Tensor((s,), "int32"),
        lengths: Tensor((s,), "int32"),
        state: Tensor((s, h, v, k), "float32"),
    ) -> tuple[Tensor((s, h, v, k), "float32"), Tensor((s,), "int32")]:
        replay_lengths = output((s,), "int32")
        with T.Kernel(s, T.ceildiv(count, 1024), threads=256) as (slot, block):
            source = T.view(initial, shape=(s, count))
            dest = T.view(state, shape=(s, count))
            partial = (accepted[slot] > 0) & (accepted[slot] < lengths[slot])
            if block == 0:
                for j in T.Parallel(1):
                    replay_lengths[slot] = T.if_then_else(partial, accepted[slot], 0)
            if partial:
                for j in T.Parallel(1024):
                    index = block * 1024 + j
                    if index < count:
                        dest[slot, index] = source[slot, index]
        return (state, replay_lengths)

    return compute


def restore_conv(p):
    import tilelang.language as T

    s, c = (p["slots"], p["chunk"])
    r = s * c
    channels = 8192

    @kernel
    def compute(
        initial: Tensor((s, channels, 3), "float32"),
        saved_x: Tensor((r, channels), "bfloat16"),
        accepted: Tensor((s,), "int32"),
        lengths: Tensor((s,), "int32"),
        state: Tensor((s, channels, 3), "float32"),
    ) -> Tensor((s, channels, 3), "float32"):
        with T.Kernel(s, channels // 256, threads=256) as (slot, tile):
            if (accepted[slot] > 0) & (accepted[slot] < lengths[slot]):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    for history in T.unroll(3):
                        pos = accepted[slot] - 3 + history
                        if pos >= 0:
                            state[slot, col, history] = T.cast(
                                saved_x[slot * c + pos, col], "float32"
                            )
                        else:
                            state[slot, col, history] = initial[slot, col, 3 + pos]
        return state

    return compute


def journal_restore(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    snapshots = chunk if p.get("full_checkpoints", False) else chunk - 1
    commit = p.get("commit", True)
    slots, chunk = (p["slots"], p["chunk"])
    slots, chunk = (p["slots"], p["chunk"])
    shape = tuple(p["shape"])
    width = 1
    for size in shape:
        width *= size

    @kernel
    def compute(
        checkpoints: Tensor((slots, snapshots, *shape), "float32"),
        state: Tensor((slots, *shape), "float32"),
        accepted: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((slots, *shape), "float32"):
        saved = T.view(checkpoints, shape=(slots, snapshots, width))
        dest = T.view(state, shape=(slots, width))
        with T.Kernel(slots, T.ceildiv(width, 256), threads=256) as (slot, tile):
            if (accepted[slot] > 0) & (
                (accepted[slot] < lengths[slot]) | (snapshots == chunk)
            ):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < width:
                        dest[slot, col] = saved[slot, accepted[slot] - 1, col]
        return state

    return compute


def mapped_scan(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, d, tile = (
        p["heads"],
        p["d"],
        p.get("webgpu_tile", p.get("tile", 32)),
    )

    @kernel
    def compute(
        q: Tensor((rows, heads, d), "float32"),
        k: Tensor((rows, heads, d), "float32"),
        v: Tensor((rows, heads, d), "float32"),
        g: Tensor((rows, heads), "float32"),
        beta: Tensor((rows, heads), "float32"),
        mapping: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((pool, heads, d, d), "float32"),
    ) -> tuple[
        Tensor((pool, heads, d, d), "float32"), Tensor((rows, heads, d), "float32")
    ]:
        out = output((rows, heads, d), "float32")
        with T.Kernel(slots, heads, d // tile, threads=128) as (batch, head, part):
            request = mapping[batch]
            if request >= 0:
                matrix = T.alloc_fragment((tile, d), "float32")
                prediction = T.alloc_fragment((tile, d), "float32")
                inner = T.alloc_fragment((tile,), "float32")
                result = T.alloc_fragment((tile,), "float32")
                qq = T.alloc_shared((d,), "float32")
                kk = T.alloc_shared((d,), "float32")
                T.copy(
                    state[request, head, part * tile : part * tile + tile, :], matrix
                )
                for t in T.serial(chunk):
                    row = batch * chunk + t
                    if t < lengths[batch]:
                        for j in T.Parallel(d):
                            qq[j] = q[row, head, j]
                            kk[j] = k[row, head, j]
                        for i, j in T.Parallel(tile, d):
                            matrix[i, j] *= T.exp(g[row, head])
                            prediction[i, j] = matrix[i, j] * kk[j]
                        T.reduce_sum(prediction, inner, dim=1)
                        for i, j in T.Parallel(tile, d):
                            matrix[i, j] += (
                                kk[j]
                                * beta[row, head]
                                * (v[row, head, part * tile + i] - inner[i])
                            )
                            prediction[i, j] = matrix[i, j] * qq[j]
                        T.reduce_sum(prediction, result, dim=1)
                        for i in T.Parallel(tile):
                            out[row, head, part * tile + i] = result[i]
                    else:
                        for i in T.Parallel(tile):
                            out[row, head, part * tile + i] = 0
                if p.get("commit", True):
                    T.copy(
                        matrix,
                        state[request, head, part * tile : part * tile + tile, :],
                    )
        return (state, out)

    return compute


def decode_recurrent(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, heads, key, value = (p[n] for n in ("slots", "heads", "key", "value"))
    threads = p.get("threads", 128)
    tile = p.get("value_tile", 32)
    if (
        any(
            (
                type(v) is not int or v <= 0
                for v in (slots, heads, key, value, tile, threads)
            )
        )
        or key & key - 1
        or tile & tile - 1
        or value % tile
        or (threads not in (64, 128, 256))
    ):
        raise ValueError("invalid gated delta schedule")

    @kernel
    def compute(
        q: Tensor((slots, heads, key), "float32"),
        k: Tensor((slots, heads, key), "float32"),
        v: Tensor((slots, heads, value), "float32"),
        log_decay: Tensor((slots, heads), "float32"),
        beta: Tensor((slots, heads), "float32"),
        active: Tensor((slots,), "int32"),
        state: Tensor((slots, heads, value, key), "float32"),
    ) -> tuple[
        Tensor((slots, heads, value, key), "float32"),
        Tensor((slots, heads, value), "float32"),
    ]:
        out = output((slots, heads, value), "float32")
        with T.Kernel(slots, heads, value // tile, threads=threads) as (
            slot,
            head,
            part,
        ):
            if active[slot] != 0:
                matrix = T.alloc_fragment((tile, key), "float32")
                prediction = T.alloc_fragment((tile, key), "float32")
                inner = T.alloc_fragment((tile,), "float32")
                result = T.alloc_fragment((tile,), "float32")
                q_shared = T.alloc_shared((key,), "float32")
                k_shared = T.alloc_shared((key,), "float32")
                for j in T.Parallel(key):
                    q_shared[j] = q[slot, head, j]
                    k_shared[j] = k[slot, head, j]
                for i, j in T.Parallel(tile, key):
                    matrix[i, j] = state[slot, head, part * tile + i, j] * T.exp(
                        log_decay[slot, head]
                    )
                    prediction[i, j] = matrix[i, j] * k_shared[j]
                T.reduce_sum(prediction, inner, dim=1)
                for i, j in T.Parallel(tile, key):
                    matrix[i, j] += (
                        k_shared[j]
                        * beta[slot, head]
                        * (v[slot, head, part * tile + i] - inner[i])
                    )
                    state[slot, head, part * tile + i, j] = matrix[i, j]
                    prediction[i, j] = matrix[i, j] * q_shared[j]
                T.reduce_sum(prediction, result, dim=1)
                for i in T.Parallel(tile):
                    out[slot, head, part * tile + i] = result[i]
            else:
                for i in T.Parallel(tile):
                    out[slot, head, part * tile + i] = 0
        return (state, out)

    return compute


def prefill_scan(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    commit = p.get("commit", True)
    slots, chunk = (p["slots"], p["chunk"])
    heads, key, value = (32, 128, 128)
    tile = p.get("value_tile", 32)

    @kernel
    def compute(
        q: Tensor((rows, heads, key), "float32"),
        k: Tensor((rows, heads, key), "float32"),
        v: Tensor((rows, heads, value), "float32"),
        g: Tensor((rows, heads), "float32"),
        beta: Tensor((rows, heads), "float32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((slots, heads, value, key), "float32"),
    ) -> tuple[
        Tensor((slots, heads, value, key), "float32"),
        Tensor((rows, heads, value), "float32"),
    ]:
        out = output((rows, heads, value), "float32")
        with T.Kernel(slots, heads, value // tile, threads=128) as (slot, head, part):
            matrix = T.alloc_fragment((tile, key), "float32")
            prediction = T.alloc_fragment((tile, key), "float32")
            inner = T.alloc_fragment((tile,), "float32")
            result = T.alloc_fragment((tile,), "float32")
            qq = T.alloc_shared((key,), "float32")
            kk = T.alloc_shared((key,), "float32")
            T.copy(state[slot, head, part * tile : part * tile + tile, :], matrix)
            for t in T.serial(chunk):
                row = slot * chunk + t
                if t < lengths[slot]:
                    for j in T.Parallel(key):
                        qq[j] = q[row, head, j]
                        kk[j] = k[row, head, j]
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] *= T.exp(g[row, head])
                        prediction[i, j] = matrix[i, j] * kk[j]
                    T.reduce_sum(prediction, inner, dim=1)
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] += (
                            kk[j]
                            * beta[row, head]
                            * (v[row, head, part * tile + i] - inner[i])
                        )
                        prediction[i, j] = matrix[i, j] * qq[j]
                    T.reduce_sum(prediction, result, dim=1)
                    for i in T.Parallel(tile):
                        out[row, head, part * tile + i] = result[i]
                else:
                    for i in T.Parallel(tile):
                        out[row, head, part * tile + i] = 0
            if commit:
                T.copy(matrix, state[slot, head, part * tile : part * tile + tile, :])
        return (state, out)

    return compute


def journal_scan(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    snapshots = chunk if p.get("full_checkpoints", False) else chunk - 1
    commit = p.get("commit", True)
    slots, chunk = (p["slots"], p["chunk"])
    heads, key, value = (32, 128, 128)
    tile = p.get("value_tile", 32)

    @kernel
    def compute(
        q: Tensor((rows, heads, key), "float32"),
        k: Tensor((rows, heads, key), "float32"),
        v: Tensor((rows, heads, value), "float32"),
        g: Tensor((rows, heads), "float32"),
        beta: Tensor((rows, heads), "float32"),
        lengths: Tensor((slots,), "int32"),
        state: Tensor((slots, heads, value, key), "float32"),
        checkpoints: Tensor((slots, snapshots, heads, value, key), "float32"),
    ) -> tuple[
        Tensor((slots, heads, value, key), "float32"),
        Tensor((rows, heads, value), "float32"),
        Tensor((slots, snapshots, heads, value, key), "float32"),
    ]:
        out = output((rows, heads, value), "float32", position=7)
        with T.Kernel(slots, heads, value // tile, threads=128) as (slot, head, part):
            matrix = T.alloc_fragment((tile, key), "float32")
            prediction = T.alloc_fragment((tile, key), "float32")
            inner = T.alloc_fragment((tile,), "float32")
            result = T.alloc_fragment((tile,), "float32")
            qq = T.alloc_shared((key,), "float32")
            kk = T.alloc_shared((key,), "float32")
            T.copy(state[slot, head, part * tile : part * tile + tile, :], matrix)
            for t in T.serial(chunk):
                row = slot * chunk + t
                if t < lengths[slot]:
                    for j in T.Parallel(key):
                        qq[j] = q[row, head, j]
                        kk[j] = k[row, head, j]
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] *= T.exp(g[row, head])
                        prediction[i, j] = matrix[i, j] * kk[j]
                    T.reduce_sum(prediction, inner, dim=1)
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] += (
                            kk[j]
                            * beta[row, head]
                            * (v[row, head, part * tile + i] - inner[i])
                        )
                        prediction[i, j] = matrix[i, j] * qq[j]
                    T.reduce_sum(prediction, result, dim=1)
                    if t < snapshots:
                        T.copy(
                            matrix,
                            checkpoints[
                                slot, t, head, part * tile : part * tile + tile, :
                            ],
                        )
                    for i in T.Parallel(tile):
                        out[row, head, part * tile + i] = result[i]
                else:
                    for i in T.Parallel(tile):
                        out[row, head, part * tile + i] = 0
            if commit:
                T.copy(matrix, state[slot, head, part * tile : part * tile + tile, :])
        return (state, out, checkpoints)

    return compute


def restore(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    snapshots = chunk if p.get("full_checkpoints", False) else chunk - 1
    shape = tuple(p["shape"])
    strides = []
    width = 1
    for size in reversed(shape):
        strides.insert(0, width)
        width *= size

    @kernel
    def compute(
        checkpoints: Tensor((slots, snapshots, *shape), "float32"),
        state: Tensor((slots, *shape), "float32"),
        accepted: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((slots, *shape), "float32"):
        with T.Kernel(slots, T.ceildiv(width, 256), threads=256) as (slot, tile):
            if (accepted[slot] > 0) & (
                (accepted[slot] < lengths[slot]) | (snapshots == chunk)
            ):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < width:
                        indices = tuple(
                            (
                                col // stride % size
                                for stride, size in zip(strides, shape)
                            )
                        )
                        state[slot, *indices] = checkpoints[
                            slot, accepted[slot] - 1, *indices
                        ]
        return state

    return compute
