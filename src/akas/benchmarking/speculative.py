"""Measure the complete proposal, verification, commit and repair token phase."""

import time

from akas.benchmarking.session import Session
from akas.requests import Request


class SpeculativeSession(Session):
    """Use the same workload accounting as AR, clipping the final draft round.

    Decode includes coordinator readbacks and external draft work. It never
    substitutes an AR target timing for speculative execution. Closing a session
    releases its requests; the caller still owns both model executors.
    """

    def __init__(self, decoder, *, batch, chunk=32, timeout=0):
        super().__init__(decoder.target, batch=batch, chunk=chunk, timeout=timeout)
        self.decoder = decoder
        self.requests = []
        self.stats_before = dict(decoder.stats)

    @property
    def allocated_bytes(self):
        executors = [self.executor]
        draft = getattr(self.decoder.proposer, "draft", None)
        if draft is not None:
            executors.append(draft)
        storage = {}
        for executor in executors:
            buffers = [*executor.weights.values(), *executor.batches.states.values()]
            for frame in executor.batches.frames.values():
                buffers.extend(frame.buffers.values())
            for buffer in buffers:
                while hasattr(buffer, "_base"):
                    buffer = buffer._base
                storage[id(buffer)] = buffer
        return sum(buffer.nbytes for buffer in storage.values())

    def reset(self):
        self.close()
        try:
            for _ in range(self.batch):
                request = Request([], 1)
                request.slot = self.decoder.admit()
                self.requests.append(request)
        except BaseException:
            self.close()
            raise
        self.stats_before = dict(self.decoder.stats)
        self.deadline = time.perf_counter() + self.timeout if self.timeout > 0 else None

    def prefill(self, prompts, *, read=True):
        if len(prompts) != len(self.requests) or any(not len(prompt) for prompt in prompts):
            raise ValueError("one nonempty prompt per admitted request is required")
        base = [len(request.prompt) for request in self.requests]
        for request, prompt in zip(self.requests, prompts):
            request.prompt.extend(map(int, prompt))
        predicted = [None] * len(prompts)
        chunk = self.decoder.prefill_geometry(self.prefill_chunk)
        for start in range(0, max(map(len, prompts)), chunk):
            self._check_timeout()
            active = [i for i, prompt in enumerate(prompts) if start < len(prompt)]
            requests = [self.requests[i] for i in active]
            for index in active:
                self.requests[index].offset = base[index] + start
            sequences = [[int(token) for token in prompts[i][start:start + chunk]] for i in active]
            result = self.decoder.prefill_batch(requests, sequences, chunk)
            for index, token in zip(active, result):
                predicted[index] = int(token)
        for request, token in zip(self.requests, predicted):
            request.output = [token]
            request.offset = len(request.prompt)
        self.tokens = predicted
        self._check_timeout()
        return predicted if read else None

    def decode(self, steps, *, read=True):
        if type(steps) is not int or steps < 1:
            raise ValueError("decode steps must be positive")
        if not self.requests or any(not request.output for request in self.requests):
            raise ValueError("benchmark decode requires prefill")
        for request in self.requests:
            if self.executor.positions[request.slot] + steps > self.capacity:
                raise ValueError("context exhausted")
            request.limit = len(request.output) + steps
        while active := [request for request in self.requests if len(request.output) < request.limit]:
            self._check_timeout()
            outputs = self.decoder.decode_many(active)
            if len(outputs) != len(active) or any(not output for output in outputs):
                raise ValueError("speculative decoder returned an incomplete batch")
            for request, output in zip(active, outputs):
                if len(request.output) + len(output) > request.limit:
                    raise ValueError("speculative decoder exceeded the output budget")
                request.output.extend(output)
        self._check_timeout()
        self.tokens = [request.output[-1] for request in self.requests]
        return self.tokens if read else None

    @property
    def stats(self):
        return {key: value - self.stats_before[key] for key, value in self.decoder.stats.items()}

    def close(self):
        for request in reversed(self.requests):
            self.decoder.release(request.slot)
        self.requests.clear()
        self.tokens, self.deadline = None, None
