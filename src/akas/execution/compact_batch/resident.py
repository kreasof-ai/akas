"""Resident Qwen dense resources and persistent request-slot lifecycle."""

from __future__ import annotations
import ctypes as ct
from pathlib import Path
from threading import get_ident
import numpy as np
from akas.memory.layouts import allocate_hybrid_states
from akas.artifacts.fingerprints import compact_batch_hashes as implementation
from akas.execution.compact_batch.batch import ExecutionBatch
from akas.models.contracts import Capabilities


class BatchedModel:
    capabilities = Capabilities(native_batch=True)

    def __init__(
        self,
        checkpoint,
        bundle,
        device,
        *,
        capacity=256,
        context=4096,
        fused_head=False,
    ):
        self.device = device
        self.thread = get_ident()
        self.capacity = capacity
        self.context = context
        self.fused_head = fused_head
        self.implementation = implementation()
        self.pending = {}
        self.closed = False
        self.generation = device._generation
        self.checkpoint = self.checkpoint_type(checkpoint)
        self.config = self.checkpoint.config
        self.architecture = self.definition(self.config, self.checkpoint)
        self.positions = np.zeros(capacity, dtype="int32")
        self.occupied = np.zeros(capacity, dtype=bool)
        self.weights = self.checkpoint.upload(device)
        self.states = {}
        self.batches = {}
        self.bundle = Path(bundle)
        self.device.driver.lib.cuMemsetD8_v2.argtypes = [
            ct.c_uint64,
            ct.c_ubyte,
            ct.c_size_t,
        ]
        self.device.driver.lib.cuMemsetD8_v2.restype = ct.c_int
        self.states = allocate_hybrid_states(device, self.config, capacity, context)

    def _check(self):
        if self.closed or self.generation != self.device._generation:
            raise RuntimeError("dense engine is closed or its device session expired")
        if get_ident() != self.thread:
            raise RuntimeError("dense engine must run on its owner thread")
        self.device._check()

    def admit(self):
        self._check()
        free = np.flatnonzero(~self.occupied)
        if not free.size:
            raise ValueError("request capacity exhausted")
        request = int(free[0])
        self.positions[request] = 0
        # Only recurrent state must be zeroed. KV beyond position is never read.
        for layer, kind in enumerate(self.config.layers):
            if kind == "linear_attention":
                for buffer in self.states[layer]:
                    size = buffer.nbytes // self.capacity
                    self.device.driver.call(
                        "cuMemsetD8_v2", buffer.pointer + request * size, 0, size
                    )
        self.occupied[request] = True
        return request

    def release(self, request):
        self._check()
        if not 0 <= request < self.capacity or not self.occupied[request]:
            raise ValueError("request is not owned")
        if request in self.pending:
            raise ValueError(
                "discard pending verification before releasing its request"
            )
        if hasattr(self, "deferred"):
            self.deferred.forget(request)
        self.occupied[request] = False
        self.positions[request] = 0

    def batch(self, count, chunk=1, *, verify=False):
        self._check()
        if not 1 <= count <= self.capacity:
            raise ValueError("invalid active count")
        slots = 1 << (count - 1).bit_length()
        key = (slots, chunk, verify)
        if key not in self.batches:
            path = (
                self.bundle
                / f"{'mtp-' if getattr(self, 'draft', False) else ''}inference-s{slots}-t{chunk}{'-verify' if verify else ''}.json"
            )
            self.batches[key] = ExecutionBatch(self, path)
        return self.batches[key]

    def decode(self, requests, tokens):
        return self.batch(len(requests)).run(requests, [[token] for token in tokens])

    def prefill(self, requests, prompts, *, chunk=32):
        predicted = [None] * len(requests)
        for begin in range(0, max(map(len, prompts)), chunk):
            active = [i for i, prompt in enumerate(prompts) if begin < len(prompt)]
            for group in range(0, len(active), 32):
                indices = active[group : group + 32]
                batch = self.batch(len(indices), chunk)
                result = batch.run(
                    [requests[i] for i in indices],
                    [prompts[i][begin : begin + chunk] for i in indices],
                )
                for i, token in zip(indices, result):
                    predicted[i] = token
        return predicted

    def close(self):
        if self.closed:
            return
        self._check()
        for batch in self.batches.values():
            batch.close()
        if hasattr(self, "deferred"):
            self.deferred.close()
        for state in self.states.values():
            for b in state:
                b.release()
        for w in self.weights.values():
            w.release()
        self.closed = True
