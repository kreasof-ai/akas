"""Portable suffix gathers and controls for ragged request chunks."""


def make_kernel(kind, p):
    import tilelang.language as T

    rows, chunk, tail = p["r"], p["chunk"], p["tail"]
    if not 0 < tail <= chunk:
        raise ValueError("suffix length must be within the input chunk")
    if kind == "chunk_tail":
        width = p["c"]

        @T.prim_func
        def gather(x: T.Tensor((rows * chunk * width,), "float32"),
                   lengths: T.Tensor((rows,), "int32"),
                   out: T.Tensor((rows * tail * width,), "float32")):
            with T.Kernel(T.ceildiv(tail * width, 64), rows, threads=64) as (block, row):
                index = block * 64 + T.get_thread_binding()
                if index < tail * width:
                    token = index // width
                    if token < T.min(lengths[row], tail):
                        out[row * tail * width + index] = x[
                            (row * chunk + T.max(lengths[row] - tail, 0) + token) * width + index % width]
                    else:
                        out[row * tail * width + index] = 0

        return gather
    if kind == "tail_controls":

        @T.prim_func
        def controls(positions: T.Tensor((rows * chunk,), "int32"),
                     lengths: T.Tensor((rows,), "int32"),
                     outpositions: T.Tensor((rows * tail,), "int32"),
                     outactive: T.Tensor((rows * tail,), "int32"),
                     outlengths: T.Tensor((rows,), "int32")):
            with T.Kernel(T.ceildiv(tail, 64), rows, threads=64) as (block, row):
                token = block * 64 + T.get_thread_binding()
                if token < tail:
                    if token < T.min(lengths[row], tail):
                        outpositions[row * tail + token] = positions[
                            row * chunk + T.max(lengths[row] - tail, 0) + token]
                        outactive[row * tail + token] = 1
                    else:
                        outpositions[row * tail + token] = 0
                        outactive[row * tail + token] = 0
                if token == 0:
                    outlengths[row] = T.min(lengths[row], tail)

        return controls
    raise ValueError("unsupported suffix operation: " + kind)


def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    rows, chunk, tail = p["r"], p["chunk"], p["tail"]
    values["lengths"][:] = np.resize(np.asarray([chunk, 0, min(chunk, 2), max(0, chunk - 1)], dtype="int32"), rows)
    if "positions" in values:
        values["positions"][:] = np.arange(rows * chunk, dtype="int32") + 17
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "chunk_tail":
        inputs = values["x"].reshape(rows, chunk, p["c"])
        out = np.zeros((rows, tail, p["c"]), dtype="float32")
        for row, length in enumerate(values["lengths"]):
            count = min(length, tail)
            out[row, :count] = inputs[row, max(length - tail, 0):length]
        expected["out"][:] = out.reshape(values["out"].shape)
    else:
        outpositions = np.zeros((rows, tail), dtype="int32")
        outactive = np.zeros((rows, tail), dtype="int32")
        positions = values["positions"].reshape(rows, chunk)
        for row, length in enumerate(values["lengths"]):
            count = min(length, tail)
            outpositions[row, :count] = positions[row, max(length - tail, 0):length]
            outactive[row, :count] = 1
        expected["outpositions"][:] = outpositions.reshape(values["outpositions"].shape)
        expected["outactive"][:] = outactive.reshape(values["outactive"].shape)
        expected["outlengths"][:] = np.minimum(values["lengths"], tail)
    return ValidationCase(values, expected, rtol=0, atol=0)
