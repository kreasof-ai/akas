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
