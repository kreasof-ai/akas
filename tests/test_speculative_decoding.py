"""Proposal heads share acceptance and protected-prefix commit semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from akas.generation.speculative.decoding import SpeculativeDecoder


class Target:
    def __init__(self):
        self.capacity, self.context = 4, 32
        self.config = SimpleNamespace(vocab=100)
        self.positions = np.zeros(4, dtype="int32")
        self.pending = None

    def verify(self, slots, sequences, *, chunk):
        assert self.pending is None
        self.pending = slots
        return [[(token + 1) % 100 for token in sequence] for sequence in sequences]

    def commit(self, slots, counts):
        assert self.pending == slots
        self.positions[slots] += counts
        self.pending = None

    def discard(self, slots):
        assert self.pending == slots
        self.pending = None


class FixtureHead:
    def __init__(self, proposals):
        self.proposals = proposals
        self.repairs = []

    def propose(self, requests, limits, *, target):
        return [proposal[:limit] for proposal, limit in zip(self.proposals, limits)]

    def repair(self, requests, outputs, positions, *, target, chunk):
        self.repairs.append((outputs, positions))


def requests():
    return [SimpleNamespace(slot=3, output=[3], limit=12),
            SimpleNamespace(slot=1, output=[8], limit=12)]


def test_generic_head_acceptance_and_mismatch_share_one_commit_flow():
    target = Target()
    head = FixtureHead([[4, 5, 6], [9, 55, 56]])
    decoder = SpeculativeDecoder(target, head)
    result = decoder.decode_many(requests())
    assert result == [[4, 5, 6, 7], [9, 10]]
    np.testing.assert_array_equal(target.positions, [0, 2, 0, 4])
    assert head.repairs == [(result, [0, 0])]
    assert target.pending is None
    assert decoder.stats == dict(rounds=1, request_rounds=2, proposed=6, accepted=4, emitted=6)


def test_request_limits_bound_proposals_and_bonus_tokens():
    target = Target()
    head = FixtureHead([[4, 5, 6], [9, 10, 11]])
    decoder = SpeculativeDecoder(target, head)
    batch = requests()
    batch[0].limit = 2
    batch[1].limit = 3
    assert decoder.decode_many(batch) == [[4], [9, 10]]
    np.testing.assert_array_equal(target.positions, [0, 2, 0, 1])


def test_incomplete_verification_discards_pending_state():
    target = Target()
    verify = target.verify
    target.verify = lambda *args, **kwargs: verify(*args, **kwargs)[:1]
    decoder = SpeculativeDecoder(target, FixtureHead([[4], [9]]))
    with pytest.raises(ValueError, match="incomplete batch"):
        decoder.decode_many(requests())
    assert target.pending is None
    np.testing.assert_array_equal(target.positions, 0)


@pytest.mark.parametrize("proposals,expected", [([[], []], 1), ([[4], [9]], 2)])
def test_filtered_proposals_use_smallest_qualified_verifier(proposals, expected):
    target = Target()
    records = [dict(phase="verify", rows=4, chunk=chunk) for chunk in (1, 2, 4)]
    selected, verified, repaired = [], [], []

    def select(count, tokens, *, limit, chunks=None):
        choices = [r["chunk"] for r in records if r["chunk"] >= tokens
                   and r["chunk"] <= limit and (chunks is None or r["chunk"] in chunks)]
        value = min(choices)
        selected.append(value)
        return SimpleNamespace(chunk=value)

    target.batches = SimpleNamespace(records=records, select_window=select)
    original = target.verify
    def verify(*args, **kwargs):
        verified.append(kwargs["chunk"])
        return original(*args, **kwargs)
    target.verify = verify
    head = FixtureHead(proposals)
    head.repair = lambda *args, **kwargs: repaired.append(kwargs["chunk"])
    result = SpeculativeDecoder(target, head, window=4).decode_many(requests())
    assert selected == [4, expected]
    assert verified == repaired == [expected]
    assert result == ([[4], [9]] if expected == 1 else [[4, 5], [9, 10]])


def test_filtered_verification_keeps_shared_head_geometry_constraints():
    target = Target()
    records = [dict(phase="verify", rows=4, chunk=chunk) for chunk in (1, 2, 4)]
    records.append(dict(phase="proposal", rows=4, chunk=4))
    target.batches = SimpleNamespace(records=records)
    def select(count, tokens, *, limit, chunks):
        assert chunks == {4}
        return SimpleNamespace(chunk=4)
    target.batches.select_window = select
    original = target.verify
    def verify(*args, **kwargs):
        assert kwargs["chunk"] == 4
        return original(*args, **kwargs)
    target.verify = verify
    head = FixtureHead([[], []])
    head.phase = "proposal"
    assert SpeculativeDecoder(target, head).decode_many(requests()) == [[4], [9]]


def test_captured_round_materializes_incompatible_windows_and_preserves_same_window_journals(monkeypatch):
    from akas.execution import replay as replay_module
    from akas.execution.executor import Executor
    from akas.execution.replay import CapturedRound

    monkeypatch.setattr(replay_module, "write", lambda device, buffer, values: None)

    executor = Executor(capacity=4, context=32, batches=SimpleNamespace(records=[]), reset=lambda slot: None)
    slots = [executor.admit() for _ in range(3)]
    assert slots == [0, 1, 2]

    materialized = []

    class FakeJournal:
        def __init__(self, rows, window, dirty):
            self.frame = SimpleNamespace(rows=rows, chunk=window)
            self.dirty = dirty

        def materialize(self, target_slots):
            target_slots = [slot for slot in target_slots if self.dirty[slot]]
            if not target_slots:
                return
            materialized.append((self.frame.rows, self.frame.chunk, tuple(target_slots)))
            self.dirty[target_slots] = False

    dirty_w4 = np.zeros(executor.capacity, dtype=bool)
    dirty_w2 = np.zeros(executor.capacity, dtype=bool)
    executor.journals[(1, 4)] = FakeJournal(1, 4, dirty_w4)
    executor.journals[(2, 4)] = FakeJournal(2, 4, dirty_w4)
    executor.journals[(2, 2)] = FakeJournal(2, 2, dirty_w2)

    proposer = SimpleNamespace(
        positions=np.zeros(executor.capacity, dtype="int32"),
        cached={0: 10, 1: 20, 2: 30},
    )
    engine = SimpleNamespace(
        target=executor,
        proposer=proposer,
        stats=dict(rounds=0, request_rounds=0, proposed=0, accepted=0, emitted=0),
    )

    def make_round(rows, window, accepted_counts):
        round_obj = object.__new__(CapturedRound)
        round_obj.engine = engine
        round_obj.device = None
        round_obj.rows = rows
        round_obj.window = window
        round_obj.verifier = SimpleNamespace(verification=None)
        round_obj.journal = executor.journals[(rows, window)]
        round_obj.control = None
        result = np.zeros((rows, window + 2), dtype="int32")
        for r, count in enumerate(accepted_counts):
            result[r, 0] = count
            result[r, 1 : count + 1] = np.arange(100 + r * 10, 100 + r * 10 + count)
            result[r, -1] = 900 + r
        round_obj.result = SimpleNamespace(to_numpy=lambda: result.copy())
        round_obj.graph = SimpleNamespace(launch=lambda: None)
        return round_obj

    req0 = SimpleNamespace(slot=0, output=[1], limit=16)
    req1 = SimpleNamespace(slot=1, output=[2], limit=16)

    # Round 1 (rows=2, window=4): both requests accept >1 tokens and mark shared w4 dirty flags.
    make_round(2, 4, [3, 2]).run([req0, req1])
    assert materialized == []
    assert list(dirty_w4[:2]) == [True, True]

    # Round 2 (rows=1, window=4): same window at different row count does not materialize w4.
    make_round(1, 4, [2]).run([req0])
    assert materialized == []
    assert list(dirty_w4[:2]) == [True, True]

    # Round 3 (rows=2, window=2): transition 4 -> 2 for [req0] materializes only slot 0 using 1-row w4 journal.
    make_round(2, 2, [2]).run([req0])
    assert materialized == [(1, 4, (0,))]
    assert list(dirty_w4[:2]) == [False, True]
    assert list(dirty_w2[:2]) == [True, False]

    # Round 4 (rows=2, window=4): transition 2 -> 4 for reordered [req1, req0] materializes slot 0 from w2,
    # while preserving slot 1's deferred w4 state without host materialization.
    make_round(2, 4, [1, 3]).run([req1, req0])
    assert materialized == [(1, 4, (0,)), (2, 2, (0,))]
    assert list(dirty_w2[:2]) == [False, False]
    assert list(dirty_w4[:2]) == [True, False]
