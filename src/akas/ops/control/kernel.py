"""Shared TileLang computations for control."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def spec_accept(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window = (p["slots"], p["window"])

    @kernel
    def compute(
        control: Tensor((slots, 5), "int32"),
        proposed: Tensor((slots * window,), "int32"),
        predicted: Tensor((slots * window,), "int32"),
    ) -> tuple[Tensor((slots,), "int32"), Tensor((slots, window + 2), "int32")]:
        lengths = output((slots,), "int32")
        result = output((slots, window + 2), "int32")
        with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
            for j in T.Parallel(128):
                row = block * 128 + j
                if row < slots:
                    accepted = T.alloc_var("int32")
                    accepted = T.cast(control[row, 2] > 0, "int32")
                    for t in T.serial(window - 1):
                        if (
                            (accepted == t + 1)
                            & (t + 1 < control[row, 2])
                            & (
                                predicted[row * window + t]
                                == proposed[row * window + t + 1]
                            )
                        ):
                            accepted += 1
                    lengths[row] = accepted
                    result[row, 0] = accepted
                    for t in T.serial(window):
                        result[row, t + 1] = predicted[row * window + t]
        return (lengths, result)

    return compute


def spec_draft(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window = (p["slots"], p["window"])
    depth = p["depth"]

    @kernel
    def compute(
        control: Tensor((slots, 5), "int32"),
        proposed: Tensor((slots * window,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((1,), "int32"),
    ]:
        mapping = output((slots,), "int32", position=2)
        position = output((slots,), "int32", position=3)
        tokens = output((slots,), "int32", position=5)
        head_position = output((slots,), "int32", position=6)
        head_active = output((slots,), "int32", position=7)
        mode = output((1,), "int32", position=8)
        with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
            for j in T.Parallel(128):
                row = block * 128 + j
                if row < slots:
                    mapping[row] = control[row, 0]
                    position[row] = control[row, 1] + depth - 1
                    lengths[row] = T.cast(control[row, 2] > depth + 1, "int32")
                    tokens[row] = proposed[row * window + depth]
                    head_position[row] = 0
                    head_active[row] = lengths[row]
                    if row == 0:
                        mode[0] = 1
        return (mapping, position, lengths, tokens, head_position, head_active, mode)

    return compute


def spec_proposal(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window = (p["slots"], p["window"])
    depth = p["depth"]

    @kernel
    def compute(predicted: Tensor((slots,), "int32")) -> Tensor(
        (slots * window,), "int32"
    ):
        proposed = output((slots * window,), "int32")
        with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
            for j in T.Parallel(128):
                row = block * 128 + j
                if row < slots:
                    proposed[row * window + depth + 1] = predicted[row]
        return proposed

    return compute


def spec_repair(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window = (p["slots"], p["window"])
    width = p["width"]

    @kernel
    def compute(
        control: Tensor((slots, 5), "int32"),
        result: Tensor((slots, window + 2), "int32"),
        target_hidden: Tensor((slots * window, width), "bfloat16"),
    ) -> tuple[
        Tensor((slots * window, width), "bfloat16"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots * window,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((1,), "int32"),
    ]:
        hidden = output((slots * window, width), "bfloat16")
        mapping = output((slots,), "int32")
        position = output((slots,), "int32")
        lengths = output((slots,), "int32")
        tokens = output((slots * window,), "int32")
        head_position = output((slots,), "int32")
        head_active = output((slots,), "int32")
        mode = output((1,), "int32")
        with T.Kernel(slots, threads=128) as row:
            for t, col in T.Parallel(window, width):
                hidden[row * window + t, col] = target_hidden[row * window + t, col]
            for t in T.Parallel(window):
                tokens[row * window + t] = result[row, t + 1]
            for j in T.Parallel(1):
                mapping[row] = control[row, 0]
                position[row] = control[row, 1]
                lengths[row] = result[row, 0]
                head_position[row] = 0
                head_active[row] = T.cast(result[row, 0] > 0, "int32")
                if row == 0:
                    mode[0] = 0
        return (
            hidden,
            mapping,
            position,
            lengths,
            tokens,
            head_position,
            head_active,
            mode,
        )

    return compute


def spec_result(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window = (p["slots"], p["window"])

    @kernel
    def compute(predicted: Tensor((slots,), "int32")) -> Tensor(
        (slots, window + 2), "int32"
    ):
        result = output((slots, window + 2), "int32")
        with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
            for j in T.Parallel(128):
                row = block * 128 + j
                if row < slots:
                    result[row, window + 1] = predicted[row]
        return result

    return compute


def spec_setup(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, window = (p["slots"], p["window"])

    @kernel
    def compute(
        control: Tensor((slots, 5), "int32"),
    ) -> tuple[
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots,), "int32"),
        Tensor((slots * window,), "int32"),
        Tensor((slots * window,), "int32"),
        Tensor((slots * window,), "int32"),
    ]:
        mapping = output((slots,), "int32")
        position = output((slots,), "int32")
        lengths = output((slots,), "int32")
        tokens = output((slots * window,), "int32")
        head_position = output((slots * window,), "int32")
        head_active = output((slots * window,), "int32")
        with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
            for j in T.Parallel(128):
                row = block * 128 + j
                if row < slots:
                    mapping[row] = control[row, 0]
                    position[row] = control[row, 1]
                    lengths[row] = control[row, 2]
                    for t in T.serial(window):
                        tokens[row * window + t] = T.if_then_else(
                            t == 0, control[row, 3], control[row, 4]
                        )
                        head_position[row * window + t] = 0
                        head_active[row * window + t] = T.cast(
                            t < control[row, 2], "int32"
                        )
        return (mapping, position, lengths, tokens, head_position, head_active)

    return compute


def copy_buffer(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    n, dtype = (p["n"], p["dtype"])

    @kernel
    def compute(x: Tensor((n,), dtype)) -> Tensor((n,), dtype):
        out = output((n,), dtype)
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                index = block * 256 + lane
                if index < n:
                    out[index] = x[index]
        return out

    return compute


def token_feedback(spec: KernelSpec, schedule: Mapping[str, object]):
    parameters = {**spec.parameters, **schedule}
    import tilelang.language as T

    rows = parameters["rows"]
    advance = parameters["advance"]

    @T.macro
    def update(predicted, tokens, lengths, positions):
        with T.Kernel(T.ceildiv(rows, 64), threads=64) as block:
            for lane in T.Parallel(64):
                row = block * 64 + lane
                if row < rows:
                    if lengths[row] > 0:
                        tokens[row] = predicted[row]
                        if advance:
                            positions[row] += 1

    if advance:

        @kernel
        def compute(
            predicted: Tensor(rows, "int32"),
            tokens: Tensor(rows, "int32"),
            lengths: Tensor(rows, "int32"),
            positions: Tensor(rows, "int32"),
        ) -> tuple[Tensor(rows, "int32"), Tensor(rows, "int32")]:
            with T.Kernel(T.ceildiv(rows, 64), threads=64) as block:
                for lane in T.Parallel(64):
                    row = block * 64 + lane
                    if row < rows:
                        if lengths[row] > 0:
                            tokens[row] = predicted[row]
                            if advance:
                                positions[row] += 1
            return (tokens, positions)

        return compute

    @kernel
    def compute(
        predicted: Tensor(rows, "int32"),
        tokens: Tensor(rows, "int32"),
        lengths: Tensor(rows, "int32"),
    ) -> Tensor(rows, "int32"):
        update(predicted, tokens, lengths, lengths)
        return tokens

    return compute
