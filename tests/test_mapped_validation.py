"""Custom state comparators retain local coordinates and exact pool isolation."""

import numpy as np
import pytest
from akas.kernels.contracts import ValidationCase, mapped_validation


@pytest.mark.parametrize("corrupt", [None, "selected", "unused"])
def test_mapped_comparator_gathers_local_rows_and_checks_unused_slots(monkeypatch, corrupt):
    from akas.kernels import registry

    def reference(kind, parameters, signature):
        assert signature[0]["shape"] == [2, 3]
        initial = np.arange(6, dtype="float32").reshape(2, 3)
        expected = initial.copy()
        expected[0, 1] += 5

        def compare(actual, target):
            assert actual.shape == target.shape == (2, 3)
            np.testing.assert_array_equal(actual, target)

        return ValidationCase({"state": initial}, {"state": expected},
                              comparators={"state": compare})

    monkeypatch.setattr(registry, "validation", reference)
    case = mapped_validation("example", {}, [
        dict(name="state", shape=[4, 3], dtype="float32"),
        dict(name="mapping", shape=[2], dtype="int32"),
    ], ((0,), 2, 4))
    actual = case.expected["state"].copy()
    if corrupt:
        actual[3 if corrupt == "selected" else 1, 0] += 1
        with pytest.raises(AssertionError):
            case.comparators["state"](actual, case.expected["state"])
    else:
        case.comparators["state"](actual, case.expected["state"])
