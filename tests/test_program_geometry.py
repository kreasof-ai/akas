"""Request capacity and scheduling budgets do not imply compiled tensor shapes."""

from types import SimpleNamespace

from akas.execution.executor import Executor
from akas.generation.speculative.decoding import SpeculativeDecoder


def test_prefill_and_decode_partition_requests_by_qualified_phase_rows():
    records = [dict(phase="decode", rows=2, chunk=1), dict(phase="prefill", rows=2, chunk=4)]
    executor = Executor(capacity=4, context=32, batches=SimpleNamespace(records=records), reset=lambda slot: None)
    executor.config = SimpleNamespace(vocab=16)
    slots = [executor.admit() for _ in range(4)]
    calls = []
    def execute(slots, sequences, chunk):
        assert len(slots) <= 2 and chunk in (1, 4)
        calls.append((tuple(slots), chunk))
        for slot, sequence in zip(slots, sequences):
            executor.positions[slot] += len(sequence)
        return [sequence[-1] for sequence in sequences]
    executor._execute_batch = execute
    assert executor.prefill(slots, [[1, 2, 3, 4, 5, 6]] * 4) == [6] * 4
    assert executor.positions.tolist() == [6] * 4
    assert calls == [((0, 1), 4), ((2, 3), 4)] * 2
    assert executor.decode(slots, [7] * 4) == [7] * 4
    assert calls[-2:] == [((0, 1), 1), ((2, 3), 1)]
    assert executor.positions.tolist() == [7] * 4


def test_speculative_prefill_geometry_qualifies_both_target_and_proposer():
    target = SimpleNamespace(capacity=4, context=32, config=object(), batches=SimpleNamespace(records=[
        dict(phase="decode", rows=2, chunk=1), dict(phase="prefill", rows=2, chunk=16),
        dict(phase="proposal", rows=2, chunk=1), dict(phase="proposal", rows=2, chunk=4)]))
    decoder = SpeculativeDecoder(target, SimpleNamespace(phase="proposal"))
    assert decoder.prefill_geometry(32) == 1
    target.batches.records.append(dict(phase="prefill", rows=2, chunk=4))
    assert decoder.prefill_geometry(32) == 4
