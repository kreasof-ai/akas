"""Independent numerical and state qualification for control."""


def copy_validation(kind, p, signature):
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"] = values["x"].copy()
    return ValidationCase(values, expected, rtol=0, atol=0)


def feedback_validation(kind, parameters, signature):
    from akas.kernels.contracts import ValidationCase, sample_arguments

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


def validation(kind, p, signature):
    import numpy as np
    from akas.generation.speculative.acceptance import accepted_prefix
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    slots, window = (p["slots"], p["window"])
    if "control" in values:
        values["control"][:] = np.column_stack(
            (
                np.arange(slots),
                np.arange(slots) * 3 + 1,
                np.full(slots, window),
                np.arange(slots) + 31,
                np.arange(slots) + 32,
            )
        )
        if slots > 1:
            values["control"][-1, 2] = max(1, window - 1)
    if "result" in values:
        values["result"][:, 0] = window
        if slots > 1:
            values["result"][-1, 0] = 1
    if kind == "spec_accept":
        rows = values["proposed"].reshape(slots, window)
        rows[:] = np.arange(slots)[:, None] + np.arange(window)[None, :] + 31
        predicted = values["predicted"].reshape(slots, window)
        predicted[:] = rows + 1
        if slots > 1:
            predicted[-1, 0] += 5
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "spec_setup":
        control = values["control"]
        for name, column in (("mapping", 0), ("position", 1), ("lengths", 2)):
            expected[name][:] = control[:, column]
        tokens = np.repeat(control[:, 4, None], window, axis=1)
        tokens[:, 0] = control[:, 3]
        expected["tokens"][:] = tokens.reshape(-1)
        expected["head_position"][:] = 0
        expected["head_active"][:] = (
            np.arange(window)[None, :] < control[:, 2, None]
        ).reshape(-1)
    elif kind == "spec_draft":
        control, depth = (values["control"], p["depth"])
        expected["mapping"][:] = control[:, 0]
        expected["position"][:] = control[:, 1] + depth - 1
        expected["lengths"][:] = control[:, 2] > depth + 1
        expected["tokens"][:] = values["proposed"].reshape(slots, window)[:, depth]
        expected["head_position"][:] = 0
        expected["head_active"][:] = expected["lengths"]
        expected["mode"][:] = 1
    elif kind == "spec_proposal":
        expected["proposed"].reshape(slots, window)[:, p["depth"] + 1] = values[
            "predicted"
        ]
    elif kind == "spec_accept":
        counts, _ = accepted_prefix(
            values["proposed"].reshape(slots, window),
            values["predicted"].reshape(slots, window),
            values["control"][:, 2],
        )
        expected["lengths"][:] = counts
        expected["result"][:, 0] = counts
        expected["result"][:, 1 : window + 1] = values["predicted"].reshape(
            slots, window
        )
    elif kind == "spec_repair":
        expected["hidden"][:] = values["target_hidden"]
        expected["tokens"][:] = values["result"][:, 1 : window + 1].reshape(-1)
        expected["mapping"][:] = values["control"][:, 0]
        expected["position"][:] = values["control"][:, 1]
        expected["lengths"][:] = values["result"][:, 0]
        expected["head_position"][:] = 0
        expected["head_active"][:] = values["result"][:, 0] > 0
        expected["mode"][:] = 0
    elif kind == "spec_result":
        expected["result"][:, window + 1] = values["predicted"]
    else:
        raise ValueError("unknown speculative control operation")
    return ValidationCase(values, expected, rtol=0, atol=0)
