"""Host-side reference inputs, comparators and state mapping shared by operations."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationCase:
    """Initial argument arrays and expected outputs/state, independent of a kernel.

    Every written buffer must have an expectation; immutable arguments are also
    checked to detect accidental writes. References use the actual input storage
    precision. No model-specific tuning callback is needed.
    """

    arguments: dict
    expected: dict
    rtol: float = 0.02
    atol: float = 0.002
    comparators: dict = field(default_factory=dict)


def sample_arguments(signature, *, seed=17):
    """Deterministic finite inputs; operation recipes constrain indexes/masks."""
    import math
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    rng = np.random.default_rng(seed)
    result = {}
    for spec in signature:
        if spec.get("kind", "buffer") != "buffer":
            raise ValueError("validation requires an operation scalar input recipe")
        shape, dtype = (tuple(spec["shape"]), spec["dtype"])
        if any((type(n) is not int or n <= 0 for n in shape)):
            raise ValueError("validation requires resolved positive buffer shapes")
        total = math.prod(shape)
        period = 1048573
        if dtype == "bfloat16":
            if total <= period:
                values = rng.normal(0, 0.15, shape).astype("float32")
                result[spec["name"]] = decode_bfloat16(encode_bfloat16(values))
            else:
                tile = decode_bfloat16(
                    encode_bfloat16(rng.normal(0, 0.15, period).astype("float32"))
                )
                result[spec["name"]] = np.resize(tile, shape)
        elif np.dtype(dtype).kind in "iu":
            if total <= period:
                result[spec["name"]] = rng.integers(3, 10, shape, dtype=dtype)
            else:
                tile = rng.integers(3, 10, period, dtype=dtype)
                result[spec["name"]] = np.resize(tile, shape)
        elif total <= period:
            result[spec["name"]] = rng.normal(0, 0.15, shape).astype(dtype)
        else:
            tile = rng.normal(0, 0.15, period).astype(dtype)
            result[spec["name"]] = np.resize(tile, shape)
    return result


def mapped_validation(kind, parameters, signature, mapping):
    """Validate a lifted kernel using its original operation's reference.

    Gather local state into reversed, nonadjacent pool slots and scatter the
    reference result back. Unselected slots remain exact sentinels, so the
    qualification checks both arithmetic and request isolation.
    """
    import numpy as np
    from akas.ops.planning import validation

    indexes, rows, capacity = mapping
    original = [dict(spec) for spec in signature if spec["name"] != "mapping"]
    names = []
    for index in indexes:
        spec = original[index]
        shape = list(spec["shape"])
        shape[0] = shape[0] // capacity * rows
        spec["shape"] = shape
        names.append(spec["name"])
    local = validation(kind, parameters, original)
    written_local = {
        name
        for name in local.expected
        if local.arguments[name] is not local.expected[name]
        and not np.array_equal(local.arguments[name], local.expected[name])
    }
    arguments = sample_arguments(
        [
            spec
            for spec in signature
            if spec["name"] in names or spec["name"] == "mapping"
        ]
    )
    expected = {
        name: value.copy() if name in written_local else value
        for name, value in arguments.items()
    }
    slots = np.linspace(0, capacity - 1, rows, dtype="int32")[::-1].copy()
    arguments["mapping"][:] = slots
    comparators = dict(local.comparators)
    for name in names:
        if name not in comparators:
            continue
        compare = comparators[name]
        shape = local.arguments[name].shape

        def mapped_compare(actual, reference, compare=compare, shape=shape):
            actual_pool = actual.reshape(capacity, -1)
            reference_pool = reference.reshape(capacity, -1)
            untouched = np.ones(capacity, dtype=bool)
            untouched[slots] = False
            np.testing.assert_array_equal(
                actual_pool[untouched], reference_pool[untouched]
            )
            compare(
                actual_pool[slots].reshape(shape), reference_pool[slots].reshape(shape)
            )

        comparators[name] = mapped_compare
    for name in list(local.arguments):
        if name in names:
            initial = arguments[name].reshape(capacity, -1)
            initial[slots] = local.arguments[name].reshape(rows, -1)
            if name in written_local:
                result = expected[name].reshape(capacity, -1)
                result[slots] = local.expected[name].reshape(rows, -1)
        else:
            arguments[name] = local.arguments[name]
            expected[name] = local.expected[name]
        local.arguments.pop(name, None)
        local.expected.pop(name, None)
    return ValidationCase(arguments, expected, local.rtol, local.atol, comparators)
