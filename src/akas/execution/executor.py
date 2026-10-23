"""One request lifecycle and batch interface over numerical forward bindings."""

from akas.execution.workspace import DeviceOwner
from akas.execution.state import StatePool

import numpy as np

from akas.models.contracts import Capabilities


class Executor(DeviceOwner):
    """Own request slots; bindings supply state resets and forward execution.

    A slot identifies persistent state. Bindings may execute it alone, at its
    original row, or through a row-to-slot mapping. Admission, validation and
    release do not depend on that choice.
    """

    capabilities = Capabilities(native_batch=True)
    prefill_group_size = 32

    def __init__(self, *, capacity, context, device=None, batches=None, reset=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.context = context
        self.batches = batches
        if batches is not None:
            batches.owner = self
        self._reset_state = reset
        super().__init__(device)
        self.state = StatePool(capacity)
        self.positions = self.state.positions
        self.occupied = self.state.occupied
        self.pending = self.state.pending
        self.journals = {}

    @classmethod
    def from_program(cls, checkpoint, directory, device, *, graphs=True):
        from dataclasses import asdict
        from akas.artifacts.read import read_program
        from akas.execution.batch import BatchBank
        from akas.execution.workspace import release

        program = read_program(directory)
        from tensor.runtime.cuda_target import matches_device
        compatible = (matches_device(program["target"], device.info["arch"])
                      if program["provider"] == "cuda" else program["target"] == device.info["arch"])
        if program["provider"] != device.info["provider"] or not compatible:
            raise ValueError("program requires a different compilation target")
        import json
        if program["config"] != json.loads(json.dumps(asdict(checkpoint.config))):
            raise ValueError("program belongs to a different checkpoint configuration")
        weights = checkpoint.upload(device,**program["upload_options"])
        bank = None
        try:
            bank = BatchBank(directory,device,program["phases"],weights,graphs=graphs,reset=program.get("reset"))
            executor = cls(capacity=program["max_requests"],context=program["max_context"],
                           device=device,batches=bank,reset=bank.reset)
            executor.checkpoint, executor.config = checkpoint, checkpoint.config
            from pathlib import Path
            executor.bundle, executor.program = Path(directory), program
            executor.weights = weights
            return executor
        except BaseException:
            if bank is not None:
                bank.close()
            release(buffers=weights.values())
            raise

    def _slot(self, slot):
        self.state.check(slot)

    def admit(self):
        self._check()
        free = np.flatnonzero(~self.occupied)
        if not free.size:
            raise ValueError("request capacity exhausted")
        return self.admit_at(int(free[0]))

    def admit_at(self, slot):
        self._check()
        return self.state.reserve(slot, reset=self._admit_slot)

    def release(self, slot):
        self._check()
        self.state.free(slot, release=self._release_slot)

    def run_batch(self, slots, sequences, *, chunk=1, phase=None, read=True):
        self._check()
        slots, sequences = self._validate_batch(slots, sequences, chunk)
        return self._execute_batch(slots, sequences, chunk, phase=phase, read=read)

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
        self._check()
        slots, sequences = self._validate_batch(slots, [[t] for t in tokens], 1)
        count = self._phase_rows(1)
        result = []
        for start in range(0, len(slots), count):
            result.extend(self.run_batch(slots[start:start + count], sequences[start:start + count]))
        return result

    def _phase_rows(self, chunk):
        records = getattr(self.batches, "records", ())
        phase = "decode" if chunk == 1 else "prefill"
        rows = [row["rows"] for row in records if row["phase"] == phase and row["chunk"] == chunk]
        if not rows:
            return self.capacity if chunk == 1 else min(self.prefill_group_size, self.capacity)
        if chunk > 1:
            bounded = [r for r in rows if r <= self.prefill_group_size]
            if bounded:
                return max(bounded)
        return max(rows)

    def prefill_geometry(self, requested=32):
        records = getattr(self.batches, "records", ())
        chunks = {row["chunk"] for row in records if row["phase"] in ("decode", "prefill")}
        if not chunks:
            return requested
        return min((chunk for chunk in chunks if chunk >= requested), default=max(chunks))

    def decode_steps(self, slots, tokens, steps, *, read=True):
        """Replay greedy steps with one control upload and optional final readback.

        This fixed request group is useful for throughput measurement. Request
        scheduling and EOS decisions still use decode(), one step at a time.
        """
        self._check()
        if type(steps) is not int or steps < 1:
            raise ValueError("decode steps must be positive")
        if tokens is None:
            slots = list(slots)
            if not slots or len(set(slots)) != len(slots):
                raise ValueError("invalid execution batch")
            for slot in slots:
                self._slot(slot)
                if slot in self.pending:
                    raise ValueError("request has an uncommitted verification")
            sequences = None
        else:
            slots, sequences = self._validate_batch(slots, [[token] for token in tokens], 1)
        if any(self.positions[slot] + steps > self.context for slot in slots):
            raise ValueError("context exhausted")
        self._materialize(slots)
        frame = self.batch(len(slots))
        if not getattr(frame, "feedback", False):
            raise ValueError("compiled decode phase has no device token feedback")
        if tokens is None:
            if (getattr(frame, "slots", None) != tuple(slots)
                    or not np.array_equal(getattr(frame, "expected_positions", None), self.positions[slots])
                    or not np.array_equal(getattr(frame, "expected_generations", None), self.state.generations[slots])):
                raise ValueError("device feedback does not match the current request state")
        else:
            frame.prepare(slots, sequences, self.positions)
        for _ in range(steps):
            frame.forward.run()
        self.last_frame = frame
        self.positions[slots] += steps
        frame.expected_positions = self.positions[slots].copy()
        frame.expected_generations = self.state.generations[slots].copy()
        if read:
            return frame.buffers[frame.prediction].to_numpy()[:len(slots)].astype(int).tolist()
        return None

    def batch(self, count, chunk=1, *, verify=False):
        self._check()
        if self.batches is None or type(count) is not int or not 1 <= count <= self.capacity:
            raise ValueError("invalid execution batch size")
        return self.batches.select(count, chunk, phase="verify" if verify else None)

    def prefill_batch(self, requests, sequences, chunk, *, on_chunk=None):
        return self.prefill([r.slot for r in requests], sequences,
                            chunk=self.prefill_geometry(chunk), on_chunk=on_chunk)

    def _prefill_groups(self, slots, sequences, chunk, *, check=None, on_chunk=None):
        self._check()
        slots, sequences = self._validate_batch(slots, sequences, chunk)
        result = []
        group_size = self._phase_rows(chunk)
        for start in range(0, len(slots), group_size):
            if check is not None:
                check()
            stop = start + group_size
            result.extend(
                self.run_batch(slots[start:stop], sequences[start:stop], chunk=chunk)
            )
            if on_chunk is not None:
                on_chunk(slots[start:stop], sequences[start:stop], self.last_frame)
        return result

    def prefill(self, slots, prompts, *, chunk=None, check=None, on_chunk=None):
        """Execute chunks and consume exposed features before their workspace is reused.

        on_chunk receives slots, token sequences and the actual compiled frame,
        including intermediate chunks whose vocabulary projection was removed.
        """
        self._check()
        slots, prompts = list(slots), [list(p) for p in prompts]
        if chunk is None:
            chunk = self.prefill_geometry()
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
        budget = self.prefill_geometry(chunk)
        qualified = {row["chunk"] for row in getattr(self.batches,"records", ())
                     if row["phase"] in ("decode","prefill") and row["chunk"] <= budget}
        qualified = qualified or {budget}
        predicted = [None] * len(slots)
        self.last_prefill_chunk, self.last_prefill_frame = None, None
        begin = 0
        while begin < max(map(len,prompts)):
            if check is not None:
                check()
            remaining = max(len(prompt)-begin for prompt in prompts)
            wanted = min(budget,remaining)
            chunk = min((value for value in qualified if value >= wanted),default=max(qualified))
            active = [i for i, prompt in enumerate(prompts) if begin < len(prompt)]
            state_only = [row for row in getattr(self.batches, "records", ())
                          if row["phase"] == "prefill_state" and row["chunk"] == chunk]
            intermediate = [i for i in active if begin + chunk < len(prompts[i])] if state_only else []
            if intermediate:
                state_rows = [row["rows"] for row in state_only]
                bounded = [r for r in state_rows if r <= self.prefill_group_size]
                group_size = max(bounded) if bounded else max(state_rows)
                for start in range(0, len(intermediate), group_size):
                    indexes = intermediate[start:start + group_size]
                    self._execute_batch([slots[i] for i in indexes],
                                        [prompts[i][begin:begin + chunk] for i in indexes],
                                        chunk, phase="prefill_state", read=False)
                    if on_chunk is not None:
                        on_chunk([slots[i] for i in indexes],
                                 [prompts[i][begin:begin + chunk] for i in indexes],
                                 self.last_frame)
                    if check is not None:
                        check()
            final = [i for i in active if i not in intermediate]
            if final:
                result = self._prefill_groups(
                    [slots[i] for i in final],
                    [prompts[i][begin:begin + chunk] for i in final], chunk,
                    check=check, on_chunk=on_chunk)
                for index, token in zip(final, result):
                    predicted[index] = token
                self.last_prefill_chunk = chunk
                self.last_prefill_frame = getattr(self,"last_frame",None)
            begin += chunk
        if check is not None:
            check()
        return predicted

    def verify(self, slots, sequences, *, chunk):
        self._check()
        slots,sequences = self._validate_batch(slots,sequences,chunk)
        self._materialize(slots)
        frame = self.batches.select(len(slots),chunk,phase="verify")
        if getattr(frame,"verification",None) is not None:
            raise ValueError("verification workspace already has pending requests")
        if frame.commit_forward is None:
            raise ValueError("program has no verification commit phase")
        frame.prepare(slots,sequences,self.positions)
        frame.forward.run()
        result = frame.forward.buffers[frame.prediction].to_numpy().reshape(frame.rows,chunk)
        frame.verification = (tuple(slots),tuple(map(len,sequences)))
        for slot in slots:
            self.pending[slot] = frame
        return [result[row,:len(sequence)].tolist() for row,sequence in enumerate(sequences)]

    def commit(self, slots, accepted):
        self._check()
        slots,accepted = tuple(slots),tuple(accepted)
        if not slots or len(slots) != len(accepted) or slots[0] not in self.pending:
            raise ValueError("no matching pending verification")
        frame = self.pending[slots[0]]
        expected,lengths = frame.verification
        if slots != expected or any(self.pending.get(slot) is not frame for slot in slots):
            raise ValueError("commit must cover its complete verification batch")
        if any(type(count) is not int or not 0 <= count <= length for count,length in zip(accepted,lengths)):
            raise ValueError("invalid accepted prefix")
        from akas.memory.transfer import write
        values = np.zeros(frame.rows,dtype="int32")
        values[:len(accepted)] = accepted
        write(self.device,frame.forward.buffers[frame.inputs["lengths"]],values)
        frame.commit_forward.run()
        for slot,count in zip(slots,accepted):
            self.positions[slot] += count
            del self.pending[slot]
        frame.verification = None

    def discard(self, slots):
        self._check()
        slots = tuple(slots)
        if not slots or slots[0] not in self.pending:
            raise ValueError("no matching pending verification")
        frame = self.pending[slots[0]]
        if frame.verification[0] != slots:
            raise ValueError("discard must cover its complete verification batch")
        for slot in slots:
            del self.pending[slot]
        frame.verification = None

    def generate(self, prompt, *, max_tokens=128, chat=True, tokenizer=None, sampling="device"):
        from akas.generation.greedy import generate_scheduled

        tokenizer = self.checkpoint.tokenizer if tokenizer is None else tokenizer
        return generate_scheduled(
            self, prompt, tokenizer=tokenizer, max_tokens=max_tokens, chat=chat, sampling=sampling
        )

    def _admit_slot(self, slot):
        if self._reset_state is None:
            raise NotImplementedError
        self._reset_state(slot)

    def _release_slot(self, slot):
        if self._reset_state is None:
            raise NotImplementedError
        self._reset_state(slot)
        for journal in self.journals.values():
            journal.dirty[slot] = False

    def _execute_batch(self, slots, sequences, chunk, *, phase=None, read=True):
        if self.batches is None:
            raise NotImplementedError
        self._materialize(slots)
        frame = self.batches.select(len(slots),chunk, phase=phase)
        if read:
            result = frame.run(slots,sequences,self.positions)
        else:
            frame.prepare(slots, sequences, self.positions)
            frame.forward.run()
            result = None
        self.last_frame = frame
        for slot,sequence in zip(slots,sequences):
            self.positions[slot] += len(sequence)
        return result

    def _close_resources(self):
        if self.batches is None:
            raise NotImplementedError
        for journal in self.journals.values():
            journal.close()
        self.journals.clear()
        self.batches.close()
        from akas.execution.workspace import release
        release(buffers=getattr(self, "weights", {}).values())

    def _materialize(self, slots, *, window=None):
        windows = {}
        for (rows, journal_window), journal in self.journals.items():
            if window is None or journal_window != window:
                windows.setdefault(journal_window, []).append((rows, journal))
        for candidates in windows.values():
            dirty = [slot for slot in slots if candidates[0][1].dirty[slot]]
            if not dirty:
                continue
            fit = [pair for pair in candidates if pair[0] >= len(dirty)]
            _, journal = min(fit) if fit else max(candidates)
            journal.materialize(dirty)

    def close(self):
        if self.closed:
            return
        self._check()
        self._close_resources()
        self.state.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
