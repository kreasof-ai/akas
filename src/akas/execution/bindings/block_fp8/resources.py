"""Experimental native Qwen3.5 MoE CUDA batch executor.

One checkpoint and one set of workspaces serve every slot. All request state
stays on the GPU. This initial executor uses a token-at-a-time prefill; a full
length throughput claim requires the optimized prefill and independent oracle
qualification, not just a successful decode microbenchmark."""

from __future__ import annotations
import ctypes as ct
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from akas.tensor.executor import capture
from akas.tensor.executor import submit
from akas.artifacts.kernels import load_kernels
import akas.memory.layouts as common_state
from akas.execution.bindings.block_fp8.requirements import requirements
from akas.artifacts.fingerprints import block_fp8_hashes as implementation_hashes
from akas.execution.executor import Executor
from akas.execution.bindings.block_fp8.workspace import allocate_workspace
from akas.execution.bindings.block_fp8.decode import build_decode_plan


class BlockFP8Binding(Executor):
    """Owner-thread batch executor with independent recurrent and KV slots."""

    def __init__(self, checkpoint, bundle, device, *, graphs=True, progress=None):
        self.device = device
        self.closed = False
        self.weights, self.kernels, self.buffers, self.states = {}, {}, {}, {}
        self.graph = None
        self.prefills = set()
        self.checkpoint = self.checkpoint_type(checkpoint)
        self.config = self.checkpoint.config
        self.architecture = self.definition(self.config, self.checkpoint)
        self.bundle = Path(bundle).resolve()
        value = json.loads((self.bundle / "inference.json").read_text())
        if (
            value.get("schema") != "tensor.qwen35-batch.v1"
            or value.get("implementation") != implementation_hashes()
            or value.get("config") != json.loads(json.dumps(asdict(self.config)))
            or value.get("target") != device.info["arch"]
        ):
            raise ValueError(
                "Qwen bundle configuration, source or target mismatch; rebuild it"
            )
        self.slots, self.context, self.splits = (
            value["slots"],
            value["context"],
            value["splits"],
        )
        self.kv_dtype = value.get("kv_dtype", "bfloat16")
        if self.kv_dtype not in ("bfloat16", "fp8"):
            raise ValueError("unsupported KV cache precision")
        if (
            self.slots not in (1, 2, 4, 8, 16, 32, 64)
            or not 1 <= self.context <= self.config.context_limit
        ):
            raise ValueError("unsupported Qwen batch capacity")
        super().__init__(capacity=self.slots, context=self.context, device=device)
        self.position = self.positions
        self.active = np.zeros(self.slots, dtype="int32")
        self._generation = device._generation
        try:
            expected = requirements(
                self.config,
                self.slots,
                self.context,
                splits=self.splits,
                kv_dtype=self.kv_dtype,
            )
            self.kernels = load_kernels(
                device, self.bundle, value["kernels"], requirements=expected
            )
            if progress:
                progress("uploading native checkpoint weights")
            self.weights = self.checkpoint.upload(device, pack_experts=True)
            if progress:
                progress(
                    f"uploaded {sum(b.nbytes for b in self.weights.values())} bytes"
                )
            self._allocate()
            self.plan = self._plan()
            if any(row.get("schedule") for row in value["kernels"].values()):
                from akas.execution.bindings.block_fp8.optimizations.linear_projections import (
                    replace,
                )

                replace(self, self.bundle)
            if graphs:
                self.graph = capture(
                    device,
                    self.plan,
                    resources=(
                        *self.weights.values(),
                        *self.buffers.values(),
                        *(b for state in self.states.values() for b in state),
                        *self.kernels.values(),
                    ),
                )
        except BaseException:
            self.close()
            raise

    def _allocate(self):
        allocate_workspace(self)
        self._allocate_states()

    def _allocate_states(self):
        self.states = common_state.allocate_hybrid_states(
            self.device,
            self.config,
            self.slots,
            self.context,
            kv_dtype=self.kv_dtype,
            zero_recurrent=True,
        )

    def reset_cache_profile(self, bundle):
        """Select an AOT cache profile between requests, retaining the weights.

        All slots must have been reset and prefill graphs closed. A failed
        rebuild closes the executor, since cache replacement releases the old
        allocation before creating the new one to stay within GPU capacity.
        """
        self._check()
        if self.prefills or self.position.any() or self.active.any():
            raise ValueError(
                "reset all requests and close prefill graphs before changing KV profile"
            )
        bundle = Path(bundle).resolve()
        value = json.loads((bundle / "inference.json").read_text())
        if (
            value.get("schema") != "tensor.qwen35-batch.v1"
            or value.get("implementation") != implementation_hashes()
            or value.get("config") != json.loads(json.dumps(asdict(self.config)))
            or value.get("target") != self.device.info["arch"]
            or (value["slots"], value["context"], value["splits"])
            != (self.slots, self.context, self.splits)
            or value.get("kv_dtype", "bfloat16") not in ("bfloat16", "fp8")
        ):
            raise ValueError("replacement cache profile configuration/source mismatch")
        had_graph = self.graph is not None
        if self.graph:
            self.graph.close()
            self.graph = None
        try:
            for kernel in self.kernels.values():
                kernel.release()
            self.kernels = {}
            for name in list(self.buffers):
                if name.startswith("_tune_"):
                    self.buffers.pop(name).release()
            for state in self.states.values():
                for buffer in state:
                    buffer.release()
            self.states = {}
            if hasattr(self, "_logical_plan"):
                del self._logical_plan
            self.kv_dtype = value.get("kv_dtype", "bfloat16")
            self.bundle = bundle
            expected = requirements(
                self.config,
                self.slots,
                self.context,
                splits=self.splits,
                kv_dtype=self.kv_dtype,
            )
            self.kernels = load_kernels(
                self.device, bundle, value["kernels"], requirements=expected
            )
            self._allocate_states()
            self.plan = self._plan()
            if any(row.get("schedule") for row in value["kernels"].values()):
                from akas.execution.bindings.block_fp8.optimizations.linear_projections import (
                    replace,
                )

                replace(self, bundle)
            if had_graph:
                self.graph = capture(
                    self.device,
                    self.plan,
                    resources=(
                        *self.weights.values(),
                        *self.buffers.values(),
                        *(b for state in self.states.values() for b in state),
                        *self.kernels.values(),
                    ),
                )
        except BaseException:
            self.close()
            raise
        return self

    def _plan(self):
        return build_decode_plan(self)

    def _submit(self):
        submit(self.device, self.plan)

    def _write(self, name, value):
        buffer = self.buffers[name]
        array = np.ascontiguousarray(value, dtype="int32")
        if array.shape != buffer.shape:
            raise ValueError("control buffer shape mismatch")
        self.device.driver.call(
            "cuMemcpyHtoD_v2",
            buffer.pointer,
            ct.c_void_p(array.ctypes.data),
            array.nbytes,
        )
        self.device.driver.call("cuStreamSynchronize", None)

    def forward(self, tokens, *, read_logits=False, debug=None):
        self._check()
        value = np.asarray(tokens)
        if value.shape != (self.slots,) or value.dtype.kind not in "iu":
            raise ValueError(
                "one integer token ID per slot is required; -1 marks an inactive slot"
            )
        if np.any(value < -1) or np.any(value >= self.config.vocab):
            raise ValueError("invalid Qwen token ID")
        active = (value >= 0).astype("int32")
        if np.any(self.position[active != 0] >= self.context):
            raise ValueError("Qwen context capacity exhausted")
        self._write("tokens", value)
        self._write("active", active)
        self.active = active
        return self.step(read_logits=read_logits, debug=debug)

    def step(self, *, read_logits=False, debug=None):
        self._check()
        if np.any(self.position[self.active != 0] >= self.context):
            raise ValueError("Qwen context capacity exhausted")
        if debug is not None:

            def inspect(entry):
                _, _, label = entry
                if label is not None:
                    debug(
                        label,
                        self.buffers[
                            "ffn" if label[1] == "mlp" else "normal"
                        ].to_numpy(),
                    )

            submit(self.device, self.plan, after_launch=inspect)
        elif self.graph is not None:
            self.graph.launch()
        else:
            self._submit()
        self.position += self.active
        result = self.buffers["tokens"].to_numpy()
        return (result, self.buffers["logits"].to_numpy()) if read_logits else result

    def reset(self, slots=None):
        self._check()
        indices = list(range(self.slots)) if slots is None else list(slots)
        if len(set(indices)) != len(indices) or any(
            type(i) is not int or not 0 <= i < self.slots for i in indices
        ):
            raise ValueError("invalid Qwen reset slots")
        self.device.synchronize()
        for layer, kind in enumerate(self.config.layers):
            if kind != "linear_attention":
                continue
            for buffer in self.states[layer]:
                size = buffer.nbytes // self.slots
                zeros = np.zeros(size // 4, dtype="float32")
                for index in indices:
                    self.device.driver.call(
                        "cuMemcpyHtoD_v2",
                        buffer.pointer + index * size,
                        ct.c_void_p(zeros.ctypes.data),
                        size,
                    )
                self.device.driver.call("cuStreamSynchronize", None)
        self.position[indices] = 0
        self.active[indices] = 0
        self._write("positions", self.position)
        self._write("active", self.active)

    @property
    def allocated_bytes(self):
        return sum(
            b.nbytes
            for b in (
                *self.weights.values(),
                *self.buffers.values(),
                *(b for state in self.states.values() for b in state),
            )
        )

    def _close_resources(self):
        for prefill in tuple(self.prefills):
            prefill.close()
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        for buffer in (
            *self.weights.values(),
            *self.buffers.values(),
            *(b for state in self.states.values() for b in state),
        ):
            buffer.release()
        for kernel in self.kernels.values():
            kernel.release()
        self.weights, self.buffers, self.states, self.kernels = {}, {}, {}, {}

    def _admit_slot(self, slot):
        self.reset([slot])

    def _release_slot(self, slot):
        self.reset([slot])

    def _execute_batch(self, slots, sequences, chunk):
        # These kernels use persistent slot IDs as execution rows. Padding is
        # explicit, and each sequence stops advancing when its chunk is done.
        predicted = [None] * len(slots)
        for index in range(max(map(len, sequences))):
            tokens = np.full(self.slots, -1, dtype="int32")
            for slot, sequence in zip(slots, sequences):
                if index < len(sequence):
                    tokens[slot] = sequence[index]
            result = self.forward(tokens)
            for row, (slot, sequence) in enumerate(zip(slots, sequences)):
                if index < len(sequence):
                    predicted[row] = int(result[slot])
        return predicted
