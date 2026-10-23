"""Independent numerical and state qualification for indexing."""


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    values["mapping"][:] = np.arange(p["rows"]) % p["pool"]
    if p["rows"] > 1:
        values["mapping"][-1] = -1
    expected = {name: value.copy() for name, value in values.items()}
    for request in values["mapping"]:
        if request >= 0:
            expected["state"][request] = 0
    return ValidationCase(values, expected, rtol=0, atol=0)


def feature_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = dict(values)
    rows = p["r"]
    if kind == "select_token":
        expected["out"] = (
            values["x"]
            .reshape(rows, p["chunk"], p["c"])[:, p["index"]]
            .copy()
            .reshape(-1)
        )
    elif kind == "concatenate":
        expected["out"] = np.concatenate(
            [
                values["left"].reshape(rows, p["left"]),
                values["right"].reshape(rows, p["right"]),
            ],
            axis=1,
        ).reshape(-1)
    elif kind == "replace_token":
        out = values["x"].reshape(rows, p["chunk"]).copy()
        out[:, p["index"]] = values["token"]
        expected["out"] = out.reshape(-1)
    else:
        raise ValueError("unknown feature indexing operation")
    return ValidationCase(values, expected, rtol=0, atol=0)
