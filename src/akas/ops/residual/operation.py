"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(("add",), "residual.kernel", "packed-baseline"),
    *variants(("add",), "residual.kernel", "packed", priority=0),
    *variants(
        ("add",),
        "residual.kernel",
        "packed",
        provider="webgpu",
        priority=0,
    ),
)

IMPLEMENTATIONS += tuple(
    item for provider in ("cuda", "webgpu")
    for item in variants(('add',), "residual.kernel", "reference", provider=provider)
)


def validation(kind, p, signature):
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = values["x"] + values["y"]
    return ValidationCase(values, expected, rtol=1e-6, atol=1e-7)


VALIDATIONS = {"add": validation}


# Output locations are part of the operation ABI, independent of compiler passes.
OUTPUT_ARGUMENTS = {"reference": {'add': (-1,)}}


from akas.ops.contracts import OutputABI

for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update({'add': OutputABI((-1,), ('out',))})
