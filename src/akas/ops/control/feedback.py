"""Device resident greedy token feedback for every model and provider."""


def make_kernel(kind, parameters):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

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

    arguments = [("predicted", rows, "int32"), ("tokens", rows, "int32"),
                 ("lengths", rows, "int32")]
    if advance:
        return primitive([*arguments, ("positions", rows, "int32")], update)

    @T.macro
    def tokens_only(predicted, tokens, lengths):
        update(predicted, tokens, lengths, lengths)

    return primitive(arguments, tokens_only)


def validation(kind, parameters, signature):
    from akas.ops.contracts import ValidationCase, sample_arguments

    arguments = sample_arguments(signature)
    arguments["lengths"][:] = 1
    if parameters["rows"] > 1:
        arguments["lengths"][-1] = 0
    expected = {name: values.copy() for name, values in arguments.items()}
    active = arguments["lengths"] > 0
    expected["tokens"][active] = arguments["predicted"][active]
    if parameters["advance"]:
        expected["positions"][active] += 1
    return ValidationCase(arguments, expected, rtol=0, atol=0)
