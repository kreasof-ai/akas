"""A compiled MTP head; numerical execution remains in the common batch bank."""

import numpy as np


class MTPHead:
    def __init__(self, target, *, phase="proposal"):
        if not any(record["phase"] == phase for record in target.batches.records):
            raise ValueError("program has no compiled proposal head")
        self.target, self.phase = target, phase
        self.positions = np.zeros(target.capacity, dtype="int32")
        self.cached = {}

    def admit(self, slot):
        self.positions[slot] = 0
        self.cached.pop(slot, None)

    def release(self, slot):
        self.positions[slot] = 0
        self.cached.pop(slot, None)

    def _run(self, slots, sequences, *, chunk, hidden=None):
        self.target._check()
        frame = self.target.batches.select(len(slots), chunk, phase=self.phase)
        frame.prepare(slots, sequences, self.positions)
        frame.write(frame.features["mode"], np.array([int(hidden is None)], dtype="int32"))
        if hidden is not None:
            frame.feature("hidden_input").copy_from(hidden)
        elif chunk != 1:
            raise ValueError("recursive MTP proposals require single-token execution")
        frame.forward.run()
        predictions = frame.buffers[frame.prediction].to_numpy()
        for slot, sequence in zip(slots, sequences):
            self.positions[slot] += len(sequence)
        return [int(value) for value in predictions[:len(slots)]]

    def frame(self, count, chunk):
        return self.target.batches.select(count, chunk, phase=self.phase)

    def prefill(self, requests, sequences, predictions, *, target, chunk):
        slots = [request.slot for request in requests]
        teacher = target.batches.select(len(slots), chunk)
        shifted = []
        for request, sequence, prediction in zip(requests, sequences, predictions):
            end = request.offset + len(sequence)
            following = request.prompt[end] if end < len(request.prompt) else prediction
            shifted.append([*sequence[1:], following])
        proposed = self._run(slots, shifted, chunk=chunk, hidden=teacher.feature("hidden"))
        self.cached.update(zip(slots, proposed))

    def propose(self, requests, limits, *, target):
        slots = [request.slot for request in requests]
        proposals = [[self.cached[slot]] if limit else [] for slot, limit in zip(slots, limits)]
        for depth in range(1, max(limits, default=0)):
            active = [row for row, limit in enumerate(limits) if limit > depth]
            predicted = self._run([slots[row] for row in active],
                                  [[proposals[row][-1]] for row in active], chunk=1)
            for row, token in zip(active, predicted):
                proposals[row].append(token)
        return proposals

    def repair(self, requests, outputs, positions, *, target, chunk):
        slots = [request.slot for request in requests]
        verifier = target.batches.select(len(slots), chunk, phase="verify")
        self.positions[slots] = positions
        repaired = self._run(slots, outputs, chunk=verifier.chunk, hidden=verifier.feature("hidden"))
        self.cached.update(zip(slots, repaired))

    def close(self):
        self.cached.clear()
