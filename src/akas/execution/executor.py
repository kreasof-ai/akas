"""One request lifecycle and batch interface over numerical forward bindings."""

from threading import get_ident

import numpy as np

from akas.models.contracts import Capabilities


class Executor:
    """Own request slots; bindings supply state resets and forward execution.

    A slot identifies persistent state. Bindings may execute it alone, at its
    original row, or through a row-to-slot mapping. Admission, validation and
    release do not depend on that choice.
    """

    capabilities = Capabilities(native_batch=True)
    prefill_group_size = 32

    def __init__(self, *, capacity, context, device=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.context = context
        self.device = device
        self.thread = get_ident()
        self.generation = device._generation if device is not None else None
        self.closed = False
        self.positions = np.zeros(capacity, dtype="int32")
        self.occupied = np.zeros(capacity, dtype=bool)
        self.pending = {}

    def _check(self):
        if self.closed:
            raise RuntimeError("executor is closed")
        if get_ident() != self.thread:
            raise RuntimeError("executor must run on its owner thread")
        if self.device is not None:
            self.device._check()
            if self.generation != self.device._generation:
                raise RuntimeError("executor belongs to an expired device session")

    def _slot(self, slot):
        if (
            type(slot) is not int
            or not 0 <= slot < self.capacity
            or not self.occupied[slot]
        ):
            raise ValueError("request is not owned")

    def admit(self):
        self._check()
        free = np.flatnonzero(~self.occupied)
        if not free.size:
            raise ValueError("request capacity exhausted")
        return self.admit_at(int(free[0]))

    def admit_at(self, slot):
        self._check()
        if (
            type(slot) is not int
            or not 0 <= slot < self.capacity
            or self.occupied[slot]
        ):
            raise ValueError("request slot is unavailable")
        self._admit_slot(slot)
        self.positions[slot] = 0
        self.occupied[slot] = True
        return slot

    def release(self, slot):
        self._check()
        self._slot(slot)
        if slot in self.pending:
            raise ValueError(
                "discard pending verification before releasing its request"
            )
        self._release_slot(slot)
        self.occupied[slot] = False
        self.positions[slot] = 0

    def run_batch(self, slots, sequences, *, chunk=1):
        self._check()
        slots, sequences = self._validate_batch(slots, sequences, chunk)
        return self._execute_batch(slots, sequences, chunk)

    def _validate_batch(self, slots, sequences, chunk):
        slots, sequences = list(slots), [list(s) for s in sequences]
        if type(chunk) is not int or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        if not slots or len(slots) != len(sequences):
            raise ValueError("invalid execution batch")
        for slot in slots:
            self._slot(slot)
            if slot in self.pending:
                raise ValueError("request has an uncommitted verification")
        if len(set(slots)) != len(slots):
            raise ValueError("duplicate request in one batch")
        for slot, sequence in zip(slots, sequences):
            if not 1 <= len(sequence) <= chunk:
                raise ValueError("invalid chunk length")
            if any(
                type(t) is not int or not 0 <= t < self.config.vocab for t in sequence
            ):
                raise ValueError("invalid token ID")
            if self.positions[slot] + len(sequence) > self.context:
                raise ValueError("context exhausted")
        return slots, sequences

    def decode(self, slots, tokens):
        return self.run_batch(slots, [[t] for t in tokens])

    def prefill_batch(self, requests, sequences, chunk):
        return self._prefill_groups([r.slot for r in requests], sequences, chunk)

    def _prefill_groups(self, slots, sequences, chunk):
        self._check()
        slots, sequences = self._validate_batch(slots, sequences, chunk)
        result = []
        for start in range(0, len(slots), self.prefill_group_size):
            stop = start + self.prefill_group_size
            result.extend(
                self.run_batch(slots[start:stop], sequences[start:stop], chunk=chunk)
            )
        return result

    def prefill(self, slots, prompts, *, chunk=32):
        self._check()
        slots, prompts = list(slots), [list(p) for p in prompts]
        if (
            type(chunk) is not int
            or chunk < 1
            or not slots
            or len(slots) != len(prompts)
        ):
            raise ValueError("invalid prefill batch")
        # Validate the whole input before executing any prefix.
        for slot in slots:
            self._slot(slot)
        if len(set(slots)) != len(slots) or any(
            slot in self.pending
            or not prompt
            or self.positions[slot] + len(prompt) > self.context
            or any(type(t) is not int or not 0 <= t < self.config.vocab for t in prompt)
            for slot, prompt in zip(slots, prompts)
        ):
            raise ValueError("invalid prefill sequence")
        predicted = [None] * len(slots)
        for begin in range(0, max(map(len, prompts)), chunk):
            active = [i for i, prompt in enumerate(prompts) if begin < len(prompt)]
            result = self._prefill_groups(
                [slots[i] for i in active],
                [prompts[i][begin : begin + chunk] for i in active],
                chunk,
            )
            for index, token in zip(active, result):
                predicted[index] = token
        return predicted

    def _admit_slot(self, slot):
        raise NotImplementedError

    def _release_slot(self, slot):
        raise NotImplementedError

    def _execute_batch(self, slots, sequences, chunk):
        raise NotImplementedError

    def _close_resources(self):
        raise NotImplementedError

    def close(self):
        if self.closed:
            return
        self._check()
        self._close_resources()
        self.occupied.fill(False)
        self.pending.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
