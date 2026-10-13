"""State row reset contract and independent correctness inputs."""

from akas.ops.registry import variants

IMPLEMENTATIONS = tuple(item for provider in ("cuda","webgpu")
                        for item in variants(("clear_rows",),"indexing.rows","state",
                                             provider=provider,space=(("threads",(64,128,256)),)))


def validation(kind,p,signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase,sample_arguments

    values = sample_arguments(signature)
    values["mapping"][:] = np.arange(p["rows"])%p["pool"]
    if p["rows"] > 1: values["mapping"][-1] = -1
    expected = {name:value.copy() for name,value in values.items()}
    for request in values["mapping"]:
        if request >= 0: expected["state"][request] = 0
    return ValidationCase(values,expected,rtol=0,atol=0)


VALIDATIONS = {"clear_rows":validation}
