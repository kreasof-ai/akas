"""RMS normalization and residual fusion composition across execution profiles."""


def rms_norm(
    emit,
    x,
    weight,
    out,
    *,
    rows,
    width,
    epsilon,
    residual=None,
    input_name="x",
    label=None,
):
    parameters = dict(r=rows, c=width, eps=epsilon)
    if residual is None:
        bindings = {input_name: x, "w": weight, "out": out}
        kind = "rms"
    else:
        bindings = dict(x=x, residual=residual, w=weight, out=out)
        kind = "add_rms"
    emit(kind, parameters, label=label, **bindings)
