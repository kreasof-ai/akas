"""Bound Qwen execution batch; borrows resident weights and request slots."""

from __future__ import annotations
import ctypes as ct
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from akas.execution.plan import ExecutionPlan
from akas.tensor.executor import capture
from akas.tensor.executor import submit
from akas.artifacts.kernels import load_kernels
from akas.artifacts.fingerprints import validate_implementation
from akas.execution.compact_batch.plan import build_plan
from akas.execution.compact_batch.workspace import allocate_workspace


class ExecutionBatch:
    def __init__(self, owner, manifest):
        self.owner = owner
        self.device = owner.device
        value = json.loads(Path(manifest).read_text())
        validate_implementation(value, owner.implementation)
        if (
            value["schema"] != "tensor.qwen35-dense.v1"
            or value["config"] != json.loads(json.dumps(asdict(owner.config)))
            or value["target"] != self.device.info["arch"]
            or (value["pool"], value["context"]) != (owner.capacity, owner.context)
        ):
            raise ValueError("dense execution profile mismatch")
        self.slots, self.chunk = value["slots"], value["chunk"]
        self.rows = self.slots * self.chunk
        self.verify = value.get("verify", False)
        self.draft = value.get("draft", False)
        if self.draft != getattr(owner, "draft", False):
            raise ValueError("wrong checkpoint branch")
        self.head_rows = self.rows if self.verify else self.slots
        self.kernels = {}
        self.buffers = {}
        self.views = []
        self.graph = None
        self.commit_graph = None
        self.commit_calls = ExecutionPlan(self.device)
        directory = Path(manifest).resolve().parent
        self.kernels = load_kernels(self.device, directory, value["kernels"])
        self._allocate()
        self.calls = self._plan()
        self.graph = capture(
            self.device,
            self.calls,
            resources=(
                *self.buffers.values(),
                *self.kernels.values(),
                *owner.weights.values(),
                *(b for state in owner.states.values() for b in state),
                *((owner.hidden,) if self.draft else ()),
            ),
        )
        if self.verify:
            self.commit_graph = capture(
                self.device, self.commit_calls, resources=self.graph.resources
            )

    def _allocate(self):
        return allocate_workspace(self)

    def _plan(self):
        return build_plan(self)

    def _submit(self):
        submit(self.device, self.calls)

    def _submit_commit(self):
        submit(self.device, self.commit_calls)

    def write(self, name, value):
        value = np.ascontiguousarray(value, dtype="int32")
        buffer = self.buffers[name]
        if value.shape != buffer.shape:
            raise ValueError("control shape mismatch")
        self.device.driver.call(
            "cuMemcpyHtoD_v2",
            buffer.pointer,
            ct.c_void_p(value.ctypes.data),
            value.nbytes,
        )

    def run(self, requests, sequences, *, hidden=None):
        self.owner._check()
        if self.verify and hasattr(self, "_verified"):
            raise ValueError(
                "commit or discard pending verification before reusing its workspace"
            )
        if any(request in self.owner.pending for request in requests):
            raise ValueError("request has an uncommitted verification")
        if hasattr(self.owner, "deferred"):
            self.owner.deferred.materialize(requests)
        if len(requests) > self.slots or len(requests) != len(sequences):
            raise ValueError("invalid execution batch")
        mapping = np.full(self.slots, -1, dtype="int32")
        positions = np.zeros(self.slots, dtype="int32")
        lengths = np.zeros(self.slots, dtype="int32")
        tokens = np.full((self.slots, self.chunk), -1, dtype="int32")
        for i, (request, sequence) in enumerate(zip(requests, sequences)):
            if (
                not 0 <= request < self.owner.capacity
                or not self.owner.occupied[request]
            ):
                raise ValueError("unowned request state")
            if not 1 <= len(sequence) <= self.chunk:
                raise ValueError("invalid chunk length")
            if self.owner.positions[request] + len(sequence) > self.owner.context:
                raise ValueError("context exhausted")
            if any(
                type(t) is not int or not 0 <= t < self.owner.config.vocab
                for t in sequence
            ):
                raise ValueError("invalid token ID")
            mapping[i] = request
            positions[i] = self.owner.positions[request]
            lengths[i] = len(sequence)
            tokens[i, : len(sequence)] = sequence
        if len(set(requests)) != len(requests):
            raise ValueError("duplicate request state in one batch")
        for name, value in (
            ("mapping", mapping),
            ("position", positions),
            ("lengths", lengths),
            ("tokens", tokens.reshape(-1)),
            ("head_position", np.zeros(self.head_rows, dtype="int32")),
            (
                "head_active",
                (
                    (np.arange(self.chunk)[None, :] < lengths[:, None]).reshape(-1)
                    if self.verify
                    else lengths > 0
                ).astype("int32"),
            ),
        ):
            self.write(name, value)
        if self.draft:
            if hidden is not None:
                hidden._check()
                if (
                    hidden.shape != self.buffers["hidden"].shape
                    or str(hidden.dtype) != "bfloat16"
                    or hidden.device is not self.device
                ):
                    raise ValueError("MTP requires aligned target BF16 hidden rows")
                self.device.driver.call(
                    "cuMemcpyDtoD_v2",
                    self.buffers["hidden"].pointer,
                    hidden.pointer,
                    hidden.nbytes,
                )
            elif self.chunk != 1:
                raise ValueError("recursive MTP drafting requires one token")
            self.write("mode", np.array([int(hidden is None)], dtype="int32"))
        self.device.driver.call("cuStreamSynchronize", None)
        self.graph.launch()
        predicted = self.buffers["predicted"].to_numpy()
        if self.verify:
            self._verified = (list(requests), lengths.copy())
            for request in requests:
                self.owner.pending[request] = self
            return predicted.reshape(self.slots, self.chunk)[: len(requests)].tolist()
        for request, sequence in zip(requests, sequences):
            self.owner.positions[request] += len(sequence)
        return predicted[: len(requests)].tolist()

    def commit(self, accepted):
        self.owner._check()
        if not self.verify or not hasattr(self, "_verified"):
            raise ValueError("no pending verification")
        requests, proposed = self._verified
        if len(accepted) != len(requests) or any(
            type(n) is not int or not 1 <= n <= proposed[i]
            for i, n in enumerate(accepted)
        ):
            raise ValueError("invalid accepted prefix lengths")
        lengths = np.zeros(self.slots, dtype="int32")
        lengths[: len(accepted)] = accepted
        self.write("lengths", lengths)
        self.device.driver.call("cuStreamSynchronize", None)
        self.commit_graph.launch()
        self.device.synchronize()
        for request, count in zip(requests, accepted):
            self.owner.positions[request] += count
        for request in requests:
            del self.owner.pending[request]
        del self._verified

    def discard(self):
        if not self.verify or not hasattr(self, "_verified"):
            raise ValueError("no pending verification")
        for request in self._verified[0]:
            del self.owner.pending[request]
        del self._verified

    def close(self):
        if self.commit_graph:
            self.commit_graph.close()
            self.commit_graph = None
        if self.graph:
            self.graph.close()
            self.graph = None
        for b in self.buffers.values():
            b.release()
        for k in self.kernels.values():
            k.release()
