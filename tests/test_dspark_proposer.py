"""Target features stream into the draft and accepted prefixes replace noise KV."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from akas.models.dspark.proposer import DSparkProposer


class Buffer:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.released = False

    def view(self, shape):
        return Buffer(self.values.reshape(-1)[:np.prod(shape)].reshape(shape))

    def copy_columns_from(self, source, *, offset):
        self.values[:, offset:offset + source.values.shape[1]] = source.values

    def to_numpy(self):
        return self.values.copy()

    def release(self):
        self.released = True


@pytest.fixture
def pair():
    device = object()
    target = NS(device=device, capacity=4, context=32, positions=np.zeros(4, dtype="int32"))
    config = NS(target_layers=(2, 10), width=2, feature_width=4, block_size=3, mask_token=99)
    features = {f"layer.{i}": f"layer.{i}" for i in config.target_layers}
    target.batches = NS(records=[dict(phase="prefill", features=features),
                                 dict(phase="verify", features=features)])
    teacher = NS(rows=2, chunk=3, slots=(3, 1))
    hidden = {name: Buffer(np.arange(12, dtype="float32") + index * 100)
              for index, name in enumerate(features)}
    teacher.feature = hidden.__getitem__
    target.last_frame = teacher
    target.batches.select = lambda *args, **kw: teacher
    context_input = Buffer(np.zeros(24, dtype="float32"))
    context = NS(feature=lambda name: context_input)
    proposal_tokens = Buffer(np.array([[7, 8, 9], [11, 12, 13]], dtype="int32"))
    proposal_conf = Buffer(np.array([[.95, .4, .99], [.9, .9, .9]], dtype="float32"))
    prepared, uploads, forwards = [], [], []
    proposal = NS(rows=2, chunk=3, features={"anchor_input": "anchor"},
                  prepare=lambda *args: prepared.append(args),
                  write=lambda name, values: uploads.append(values.copy()),
                  forward=NS(run=lambda: forwards.append(True)),
                  feature=lambda name: {"tokens": proposal_tokens,
                                        "confidence": proposal_conf}[name])
    draft = NS(device=device, capacity=4, context=32, config=config,
               checkpoint=NS(target=target), positions=np.zeros(4, dtype="int32"),
               batches=NS(select=lambda *args, **kw: context,
                          select_window=lambda *args, **kw: proposal),
               _validate_batch=lambda *args: None,
               close=lambda: forwards.append("closed"))
    consumed = []

    def run(slots, sequences, *, chunk, phase, read):
        consumed.append((slots, sequences, context_input.values.reshape(6, 4).copy()))
        draft.positions[slots] += list(map(len, sequences))

    draft.run_batch = run
    requests = [NS(slot=3, output=[5]), NS(slot=1, output=[10])]
    return NS(target=target, draft=draft, teacher=teacher, hidden=hidden, requests=requests,
              consumed=consumed, prepared=prepared, uploads=uploads, forwards=forwards)


def test_prefill_features_and_repair_keep_only_committed_context(pair):
    p = pair
    proposer = DSparkProposer(p.target, p.draft)
    p.target.positions[[3, 1]] = [3, 2]
    proposer.prefill(p.requests, [[1, 2, 3], [4, 5]], None, target=p.target, chunk=3)
    np.testing.assert_array_equal(p.draft.positions, [0, 2, 0, 3])
    np.testing.assert_array_equal(p.consumed[0][2],
                                  np.column_stack([v.values.reshape(6, 2) for v in p.hidden.values()]))
    assert proposer.propose(p.requests, [3, 2], target=p.target) == [[7, 8, 9], [11, 12]]
    assert p.prepared[0][1] == [[5, 99, 99], [10, 99]]
    assert p.uploads[0].tolist() == [5, 10]
    # Noise execution never moves the logical draft prefix.
    np.testing.assert_array_equal(p.draft.positions, [0, 2, 0, 3])
    for hidden in p.hidden.values():
        hidden.values += 1000
    p.target.positions[[3, 1]] += [1, 3]
    proposer.repair(p.requests, [[21], [22, 23, 24]], [3, 2], target=p.target, chunk=3)
    assert p.consumed[1][1] == [[21], [22, 23, 24]]
    np.testing.assert_array_equal(p.draft.positions, p.target.positions)
    np.testing.assert_array_equal(p.consumed[1][2], p.consumed[0][2] + 1000)
    views = list(proposer._views.values())
    proposer.close()
    proposer.close()
    assert all(view.released for view in views)
    assert p.forwards.count("closed") == 1


def test_confidence_truncates_at_first_uncertain_position_and_zero_budget_skips(pair):
    p = pair
    proposer = DSparkProposer(p.target, p.draft, confidence_threshold=.5)
    assert proposer.propose(p.requests, [0, 0], target=p.target) == [[], []]
    assert p.forwards == []
    assert proposer.propose(p.requests, [3, 1], target=p.target) == [[7], [11]]
    assert p.prepared[0][1] == [[5, 99, 99], [10]]


def test_missing_target_features_and_stale_context_fail_before_draft_execution(pair):
    p = pair
    p.target.batches.records[0]["features"] = {}
    with pytest.raises(ValueError, match="hidden features"):
        DSparkProposer(p.target, p.draft)
    p.target.batches.records[0]["features"] = {"layer.2": "x", "layer.10": "y"}
    proposer = DSparkProposer(p.target, p.draft)
    p.target.positions[3] = 1
    with pytest.raises(ValueError, match="aligned"):
        proposer.propose(p.requests, [3, 3], target=p.target)
    assert p.forwards == []


def test_streaming_prefill_consumes_every_actual_target_chunk():
    from akas.execution.executor import Executor
    from akas.generation.speculative.decoding import SpeculativeDecoder

    target = Executor(capacity=1, context=32, reset=lambda slot: None)
    target.config = NS(vocab=100)
    target.admit()
    records = [dict(phase=phase, rows=1, chunk=chunk)
               for phase, chunk in (("decode", 1), ("prefill", 3), ("prefill_state", 3))]
    live = np.zeros(3, dtype="int32")

    def select(count, chunk, *, phase=None):
        def prepare(slots, sequences, positions):
            live[:] = 0
            live[:len(sequences[0])] = sequences[0]

        def run(slots, sequences, positions):
            prepare(slots, sequences, positions)
            return [sequences[0][-1] + 1]

        return NS(chunk=chunk, run=run, prepare=prepare, forward=NS(run=lambda: None))

    target.batches = NS(records=records, select=select)
    consumed = []
    proposer = NS(stream_prefill=True, prefill=lambda requests, sequences, predictions, **kw:
                  consumed.append((sequences, live.copy(), int(target.positions[0]), kw["chunk"])))
    decoder = SpeculativeDecoder(target, proposer)
    request = NS(slot=0)
    assert decoder.prefill_batch([request], [list(range(8))], 3) == [8]
    assert [row[0] for row in consumed] == [[[0, 1, 2]], [[3, 4, 5]], [[6, 7]]]
    assert [row[1].tolist() for row in consumed] == [[0, 1, 2], [3, 4, 5], [6, 7, 0]]
    assert [row[2] for row in consumed] == [3, 6, 8]
    assert [row[3] for row in consumed] == [3, 3, 3]
