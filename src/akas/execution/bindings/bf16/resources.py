"""Resident Qwen dense resources and persistent request-slot lifecycle."""

from __future__ import annotations
import ctypes as ct
from pathlib import Path
from akas.memory.layouts import allocate_hybrid_states
from akas.artifacts.fingerprints import bf16_hashes as implementation
from akas.execution.bindings.bf16.batch import ExecutionBatch
from akas.execution.executor import Executor


class BF16Binding(Executor):
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
        super().__init__(capacity=capacity, context=context, device=device)
        self.fused_head = fused_head
        self.implementation = implementation()
        self.checkpoint = self.checkpoint_type(checkpoint)
        self.config = self.checkpoint.config
        self.architecture = self.definition(self.config, self.checkpoint)
        self.weights, self.states, self.batches = {}, {}, {}
        self.bundle = Path(bundle)
        try:
            self.weights = self.checkpoint.upload(device)
            self.device.driver.lib.cuMemsetD8_v2.argtypes = [
                ct.c_uint64,
                ct.c_ubyte,
                ct.c_size_t,
            ]
            self.device.driver.lib.cuMemsetD8_v2.restype = ct.c_int
            self.states = allocate_hybrid_states(device, self.config, capacity, context)
        except BaseException:
            self.close()
            raise

    def _admit_slot(self, slot):
        # KV beyond the position is unread; only recurrent state needs clearing.
        for layer, kind in enumerate(self.config.layers):
            if kind == "linear_attention":
                for buffer in self.states[layer]:
                    size = buffer.nbytes // self.capacity
                    self.device.driver.call(
                        "cuMemsetD8_v2", buffer.pointer + slot * size, 0, size
                    )

    def _release_slot(self, slot):
        if hasattr(self, "deferred"):
            self.deferred.forget(slot)

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

    def _execute_batch(self, slots, sequences, chunk):
        return self.batch(len(slots), chunk)._run_validated(slots, sequences)

    def _close_resources(self):
        for batch in self.batches.values():
            batch.close()
        if hasattr(self, "deferred"):
            self.deferred.close()
        for state in self.states.values():
            for buffer in state:
                buffer.release()
        for weight in self.weights.values():
            weight.release()
