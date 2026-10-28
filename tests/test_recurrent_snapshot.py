"""A replay snapshot includes every state buffer and rejects slot reuse."""

from types import SimpleNamespace
import numpy as np
import pytest
from akas.benchmarking.state import RecurrentSnapshot


class Buffer:
    def __init__(self, values):
        self.values = np.asarray(values)

    def to_numpy(self):
        return self.values


def executor():
    return SimpleNamespace(
        positions=np.array([17, 23]),
        occupied=np.array([True, True]),
        pending={},
        state=SimpleNamespace(generations=np.array([2, 4])),
        device=SimpleNamespace(info={"provider": "cuda"}, synchronize=lambda: None),
        batches=SimpleNamespace(
            states={
                "positions": Buffer([17, 23]),
                "conv": Buffer(np.arange(12).reshape(2, 6)),
                "kc": Buffer(np.arange(16).reshape(2, 8)),
                "vc": Buffer(np.arange(16, 32).reshape(2, 8)),
            }
        ),
    )


def test_snapshot_restores_positions_and_all_recurrent_buffers(monkeypatch):
    from akas.memory import transfer

    def write_many(device, updates):
        for buffer, values in updates:
            buffer.values[:] = values

    monkeypatch.setattr(transfer, "write_many", write_many)
    model = executor()
    saved = RecurrentSnapshot.capture(model)
    model.positions[:] += 9
    for buffer in model.batches.states.values():
        buffer.values[:] = -1
    saved.restore(model)
    np.testing.assert_array_equal(model.positions, [17, 23])
    for name, buffer in model.batches.states.items():
        np.testing.assert_array_equal(buffer.values, saved.buffers[name])


@pytest.mark.parametrize("change", ["owner", "generation", "occupied", "pending"])
def test_snapshot_rejects_changed_lifecycle(change):
    model = executor()
    saved = RecurrentSnapshot.capture(model)
    if change == "owner":
        model = executor()
    elif change == "generation":
        model.state.generations[0] += 1
    elif change == "occupied":
        model.occupied[0] = False
    else:
        model.pending[0] = object()
    with pytest.raises(ValueError, match="generations"):
        saved.restore(model)
