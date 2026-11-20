"""Independent NumPy selective-scan and bidirectional references."""

from akas.kernels.contracts import ValidationCase, sample_arguments


def selective_scan_validation(kind, p, signature):
    import numpy as np

    values = sample_arguments(signature)
    u = values["u"].astype("float64")
    delta = values["delta"].astype("float64")
    decay = values["decay"].astype("float64")
    b = values["b"].astype("float64")
    c = values["c"].astype("float64")
    outputs = []
    for sequence in range(p["sequences"]):
        state = np.zeros((u.shape[1], decay.shape[1]), dtype="float64")
        for frame in range(p["frames"]):
            row = sequence * p["frames"] + frame
            step, x = delta[row], u[row]
            state = np.exp(step[:, None] * decay) * state + (step * x)[:, None] * b[row]
            value = np.sum(state * c[row], axis=-1) + x * values["skip"]
            z = values["gate"][row].astype("float64")
            outputs.append(value * z / (1 + np.exp(-z)))
    expected = {key: value.copy() for key, value in values.items()}
    expected["out"][:] = np.stack(outputs)
    return ValidationCase(values, expected, rtol=2e-4, atol=2e-5)


def reverse_validation(kind, p, signature):
    values = sample_arguments(signature)
    expected = {key: value.copy() for key, value in values.items()}
    expected["out"][:] = values["x"][:, ::-1].copy()
    return ValidationCase(values, expected, rtol=0, atol=0)


def bidirectional_join_validation(kind, p, signature):
    import numpy as np

    values = sample_arguments(signature)
    original = values["original"]
    result = np.concatenate(
        (values["forward"] + original, values["backward"] + original), axis=-1
    )
    expected = {key: value.copy() for key, value in values.items()}
    expected["out"][:] = result
    return ValidationCase(values, expected, rtol=0, atol=0)


VALIDATIONS = {
    "selective_scan": selective_scan_validation,
    "mamba_reverse": reverse_validation,
    "mamba_join": bidirectional_join_validation,
}
