"""Independent numerical and state qualification for residual."""


def validation(kind, p, signature):
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = values["x"] + values["y"]
    return ValidationCase(values, expected, rtol=1e-06, atol=1e-07)
