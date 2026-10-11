"""Framework-free chunked prefill sharing a Qwen batch's resident resources."""

import ctypes as ct
import json
from pathlib import Path
import numpy as np
from akas.tensor.graph import CudaGraph
from akas.execution.plan import bind as bind_call
from akas.execution.plan import submit as submit_calls
from akas.artifacts.hybrid import identity
from akas.artifacts.hybrid import requirements
from akas.artifacts.kernels import load_kernels

from akas.execution.hybrid.prefill_plan import build_prefill_plan


class Qwen35Prefill:
    def __init__(self, model, bundle):
        model._check()
        self.model = model
        self.device = model.device
        self.closed = False
        self.graph = None
        self.buffers = {}
        self.kernels = {}
        self.plan = []
        bundle = Path(bundle).resolve()
        manifest = json.loads((bundle / "prefill.json").read_text())
        if (
            manifest["schema"] != "tensor.qwen35-prefill.v1"
            or manifest["slots"] != model.slots
            or manifest["context"] != model.context
            or manifest.get("kv_dtype", "bfloat16") != model.kv_dtype
        ):
            raise ValueError("prefill bundle capacity or KV precision mismatch")
        self.chunk = manifest["chunk"]
        self.rows = model.slots * self.chunk
        try:
            self.kernels = load_kernels(self.device, bundle, manifest["kernels"])
            self._allocate()
            self._plan()
            self.graph = CudaGraph(
                self.device,
                self._submit,
                resources=(
                    *model.weights.values(),
                    *model.buffers.values(),
                    *(b for state in model.states.values() for b in state),
                    *self.buffers.values(),
                    *self.kernels.values(),
                    *model.kernels.values(),
                ),
            )
            if hasattr(model, "prefills"):
                model.prefills.add(self)
        except BaseException:
            self.close()
            raise

    def _allocate(self):
        d = self.device
        r = self.rows
        # These activations are independent of decode workspaces; checkpoint,
        # KV and recurrent tensors are shared by both execution phases.
        skip = {
            "positions",
            "active",
            "logits",
            "att_parts",
            "att_stats",
            "experts",
            "routes",
        }
        for name, buffer in self.model.buffers.items():
            if name in skip or name.startswith("_tune_"):
                continue
            self.buffers[name] = d.empty((r, *buffer.shape[1:]), buffer.dtype)
        for name in ("flat_positions", "flat_active"):
            self.buffers[name] = d.empty((r,), "int32")
        self.buffers["lengths"] = d.empty((self.model.slots,), "int32")
        self.buffers["counts"] = d.empty((256,), "int32")
        self.buffers["routes"] = d.empty((256, r), "int32")
        self.quantized = {}
        for name in (
            "normal",
            "gdn_normal",
            "att_out",
            "shared_activation",
            "expert_activation",
        ):
            x = self.buffers[name]
            bits = d.empty(x.shape, "uint8")
            scales = d.empty((*x.shape[:-1], x.shape[-1] // 128))
            self.quantized[name] = (bits, scales)
            self.buffers["quantized_" + name] = bits
            self.buffers["scales_" + name] = scales

    def _plan(self):
        return build_prefill_plan(self)

    def _submit(self):
        submit_calls(self.device, self.plan)

    def forward(self, tokens, lengths=None, *, read_logits=False):
        self.model._check()
        if self.closed:
            raise RuntimeError("prefill is closed")
        tokens = np.asarray(tokens)
        s, c = self.model.slots, self.chunk
        if tokens.shape != (s, c) or tokens.dtype.kind not in "iu":
            raise ValueError("prefill requires [slots,chunk] token IDs")
        if lengths is None:
            lengths = np.full(s, c, "int32")
        lengths = np.asarray(lengths, dtype="int32")
        if lengths.shape != (s,) or np.any(lengths < 0) or np.any(lengths > c):
            raise ValueError("invalid prefill chunk lengths")
        mask = np.arange(c)[None, :] < lengths[:, None]
        if np.any(tokens[mask] < 0) or np.any(tokens[mask] >= self.model.config.vocab):
            raise ValueError("invalid prefill token ID")
        if np.any(self.model.position + lengths > self.model.context):
            raise ValueError("Qwen context capacity exhausted")
        data = np.where(mask, tokens, -1).astype("int32").reshape(-1)
        for name, value in (("tokens", data), ("lengths", lengths)):
            value = np.ascontiguousarray(value)
            buffer = self.buffers[name]
            self.device.driver.call(
                "cuMemcpyHtoD_v2",
                buffer.pointer,
                ct.c_void_p(value.ctypes.data),
                value.nbytes,
            )
        self.device.driver.call("cuStreamSynchronize", None)
        self.model.active = (lengths > 0).astype("int32")
        self.model._write("active", self.model.active)
        self.graph.launch()
        self.model.position += lengths
        result = self.model.buffers["tokens"].to_numpy()
        return (
            (result, self.model.buffers["logits"].to_numpy()) if read_logits else result
        )

    def close(self):
        if self.closed:
            return
        if self.graph:
            self.graph.close()
            self.graph = None
        for buffer in self.buffers.values():
            buffer.release()
        for kernel in self.kernels.values():
            kernel.release()
        self.buffers, self.kernels = {}, {}
        self.closed = True
        if hasattr(self.model, "prefills"):
            self.model.prefills.discard(self)
