"""Linear operation composition over native AOT kernels; no model names."""


def linear(emit, x, weight, out, *, rows, scales=None, activation_scales=None):
    columns, depth = weight.shape
    parameters = dict(r=rows, k=depth, o=columns)
    bindings = dict(x=x, w=weight, out=out)
    if scales is not None:
        bindings["scales"] = scales
        if activation_scales is not None:
            bindings["activation_scales"] = activation_scales
    elif activation_scales is not None:
        raise ValueError("activation scales require FP8 weight scales")
    emit("bf16_linear" if scales is None else "fp8_linear", parameters, **bindings)


def packed_linear(emit, inputs, weight, out, *, rows, shape, quantization, tile):
    columns, depth = shape
    parameters = dict(r=rows, k=depth, o=columns, type=quantization, **tile)
    arguments = inputs("linear", parameters)
    emit("linear", parameters, *arguments, weight, out)


def split_linear(emit, device, x, weight, out, *, rows, parameters, scratch, views):
    """Emit split/merge calls, retaining a borrowed view for singleton splits."""
    if parameters["parts"] == 1:
        from tensor.runtime.buffer import Buffer

        scratch = Buffer(
            device, out.pointer, (rows, 1, weight.shape[0]), out.dtype, owner=out
        )
        views.append(scratch)
    emit("split_linear", parameters, x, weight, scratch)
    if parameters["parts"] > 1:
        emit("split_merge", parameters, scratch, out)
