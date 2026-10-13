"""One speculative decoding coordinator for any proposal implementation."""

import numpy as np

from akas.generation.speculative.acceptance import accepted_prefix
from akas.models.contracts import Capabilities


class SpeculativeDecoder:
    capabilities = Capabilities(native_batch=True, speculative=True)

    def __init__(self, target, proposer, *, window=4, replay=None, capture=False, deferred=False):
        if type(window) is not int or window < 2:
            raise ValueError("speculative window must contain at least two tokens")
        self.target, self.proposer = target, proposer
        self.window, self.replay = window, replay
        self.capture, self.rounds = capture, {}
        if deferred and not capture:
            raise ValueError("deferred state requires a captured greedy round")
        self.deferred = deferred
        self.capacity, self.context, self.config = target.capacity, target.context, target.config
        self.stats = dict(rounds=0, request_rounds=0, proposed=0, accepted=0, emitted=0)

    def admit(self):
        slot = self.target.admit()
        try:
            self.proposer.admit(slot)
        except BaseException:
            self.target.release(slot)
            raise
        return slot

    def release(self, slot):
        self.target.release(slot)
        self.proposer.release(slot)

    def prefill_batch(self, requests, sequences, chunk):
        group_size = getattr(self.target, "prefill_group_size", len(requests))
        result = []
        for begin in range(0, len(requests), group_size):
            group = requests[begin:begin+group_size]
            tokens = sequences[begin:begin+group_size]
            predictions = self.target.prefill_batch(group, tokens, chunk)
            self.proposer.prefill(group, tokens, predictions, target=self.target, chunk=chunk)
            result.extend(predictions)
        return result

    def decode_many(self, requests):
        if self.capture:
            from akas.execution.replay import CapturedRound
            rows = self.target.batch(len(requests), self.window, verify=True).rows
            if rows not in self.rounds:
                self.rounds[rows] = CapturedRound(self,len(requests))
            return self.rounds[rows].run(requests)
        if self.replay is not None:
            return self.replay.run(requests)
        slots = [request.slot for request in requests]
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("invalid speculative request batch")
        limits = [min(self.window, request.limit-len(request.output),
                      self.context-int(self.target.positions[request.slot]))
                  for request in requests]
        if any(limit < 1 or not request.output for request, limit in zip(requests, limits)):
            raise ValueError("request cannot decode")
        proposals = self.proposer.propose(requests, [limit-1 for limit in limits], target=self.target)
        if len(proposals) != len(requests) or any(len(row) > limit-1 for row, limit in zip(proposals, limits)):
            raise ValueError("proposer exceeded the verification budget")
        sequences = [[request.output[-1], *proposal] for request, proposal in zip(requests, proposals)]
        positions = [int(self.target.positions[slot]) for slot in slots]
        inputs = np.zeros((len(slots), self.window), dtype="int32")
        verified = np.zeros_like(inputs)
        lengths = np.array(list(map(len, sequences)), dtype="int32")
        predictions = self.target.verify(slots, sequences, chunk=self.window)
        try:
            if len(predictions) != len(sequences):
                raise ValueError("verifier returned an incomplete batch")
            for row, (sequence, prediction) in enumerate(zip(sequences, predictions)):
                if len(prediction) != len(sequence):
                    raise ValueError("verifier returned an incomplete token sequence")
                inputs[row, :len(sequence)] = sequence
                verified[row, :len(sequence)] = prediction
            counts, outputs = accepted_prefix(inputs, verified, lengths)
            self.target.commit(slots, counts.tolist())
        except BaseException:
            self.target.discard(slots)
            raise
        self.proposer.repair(requests, outputs, positions, target=self.target, chunk=self.window)
        self.stats["rounds"] += 1
        self.stats["request_rounds"] += len(slots)
        self.stats["proposed"] += sum(map(len, proposals))
        self.stats["accepted"] += int(np.sum(counts-1))
        self.stats["emitted"] += int(np.sum(counts))
        return outputs

    def close(self):
        for replay in self.rounds.values():
            replay.close()
        self.rounds.clear()
        if self.replay is not None:
            self.replay.close()
        self.proposer.close()
        self.target.close()
