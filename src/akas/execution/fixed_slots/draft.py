"""Experimental native MTP drafter sharing the target's embedding and head.

This executes the official draft layer, not a speculative acceptance engine.
Callers must initialize its shifted-token/target-hidden cache over the prefix.
Target verification and recurrent rollback remain separate required work."""

from dataclasses import asdict
from dataclasses import replace as config_replace
import json
from pathlib import Path
from threading import get_ident
import numpy as np
from akas.tensor.graph import CudaGraph
from akas.tensor.executor import bind as bind_call
from akas.execution.fixed_slots.resident import FixedSlotModel as Qwen35Batch
from akas.artifacts.kernels import identity
from akas.artifacts.kernels import load_kernels
from akas.artifacts.fingerprints import (
    fixed_slots_hashes as mtp_implementation_hashes,
)
from akas.execution.fixed_slots.prefill import Qwen35Prefill
from akas.artifacts.fingerprints import fixed_slots_hashes


class Qwen35MTP(Qwen35Batch):
    """One native draft layer with private KV; target weights remain borrowed."""

    def __init__(self, target, bundle, *, graphs=True):
        target._check()
        self.target = target
        self.device, self.thread = target.device, get_ident()
        self._generation = self.device._generation
        self.closed = False
        self.weights, self.kernels, self.buffers, self.states = {}, {}, {}, {}
        self.owned_weights = {}
        self.graph = None
        self.prefills = set()
        self.checkpoint_type = target.checkpoint_type
        self.definition = target.definition
        self.checkpoint = self.checkpoint_type(
            target.checkpoint.directory, branch="mtp"
        )
        self.config = config_replace(target.config, layers=("full_attention",))
        self.architecture = self.definition(self.config, self.checkpoint)
        self.slots, self.context, self.splits = (
            target.slots,
            target.context,
            target.splits,
        )
        self.kv_dtype = target.kv_dtype
        self.position = np.zeros(self.slots, dtype="int32")
        self.active = np.zeros(self.slots, dtype="int32")
        self.bundle = Path(bundle).resolve()
        value = json.loads((self.bundle / "inference.json").read_text())
        if (
            value.get("schema") != "tensor.qwen35-batch.v1"
            or value.get("draft") != "qwen35-mtp"
            or value.get("implementation") != mtp_implementation_hashes()
            or value.get("config") != json.loads(json.dumps(asdict(self.config)))
            or value.get("target_config")
            != json.loads(json.dumps(asdict(target.config)))
            or value.get("target") != self.device.info["arch"]
            or (
                value.get("slots"),
                value.get("context"),
                value.get("splits"),
                value.get("kv_dtype"),
            )
            != (self.slots, self.context, self.splits, self.kv_dtype)
        ):
            raise ValueError(
                "MTP bundle source, configuration or target mismatch; rebuild it"
            )
        try:
            self.kernels = load_kernels(self.device, self.bundle, value["kernels"])
            self.owned_weights = self.checkpoint.upload(self.device, pack_experts=True)
            for name, buffer in self.owned_weights.items():
                alias = name.replace("mtp.layers.0.", "model.language_model.layers.0.")
                if name == "mtp.norm.weight":
                    alias = "model.language_model.norm.weight"
                self.weights[alias] = buffer
            for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
                self.weights[name] = target.weights[name]
            self._allocate()
            self.buffers["mtp_hidden"] = self.device.empty(
                (self.slots, 2048), "bfloat16"
            )
            self.buffers["mtp_join"] = self.device.empty((self.slots, 4096), "bfloat16")
            self.buffers["mtp_fc"] = self.device.empty((self.slots, 2048), "float32")
            base = Qwen35Batch._plan(self)
            prefix = []

            def bind(kind, p, *args):
                kernel = self.kernels[identity(kind, p)]
                prefix.append(
                    (
                        kernel,
                        bind_call(self.device, kernel, args),
                        None,
                    )
                )

            b = self.buffers
            bind(
                "mtp_join",
                dict(r=self.slots, c=2048, eps=self.config.epsilon),
                b["residual"],
                b["mtp_hidden"],
                self.weights["mtp.pre_fc_norm_embedding.weight"],
                self.weights["mtp.pre_fc_norm_hidden.weight"],
                b["mtp_join"],
            )
            bind(
                "bf16_linear",
                dict(r=self.slots, k=4096, o=2048),
                b["mtp_join"],
                self.weights["mtp.fc.weight"],
                b["mtp_fc"],
            )
            bind("mtp_cast", dict(r=self.slots, c=2048), b["mtp_fc"], b["residual"])
            self.plan = [base[0], *prefix, *base[1:]]
            from akas.execution.fixed_slots.optimizations.linear_projections import (
                replace,
            )

            replace(self, self.bundle)
            if graphs:
                self.graph = CudaGraph(
                    self.device,
                    self._submit,
                    resources=(
                        *self.weights.values(),
                        *self.buffers.values(),
                        *(b for state in self.states.values() for b in state),
                        *self.kernels.values(),
                    ),
                )
            target.prefills.add(self)
        except BaseException:
            self.close()
            raise

    def _check(self):
        super()._check()
        self.target._check()

    def draft(self, tokens, hidden, *, read_logits=False):
        """Advance consecutive MTP positions using next-token IDs and target states.

        ``hidden`` is the target's final normalized BF16 hidden state for the
        preceding token. Pass a device buffer to avoid a GPU/CPU round trip.
        Prefix initialization begins at position zero; skipping directly to a
        long context would leave unreadable entries in the draft KV cache.
        """
        self._check()
        dest = self.buffers["mtp_hidden"]
        if (
            getattr(hidden, "shape", None) != dest.shape
            or str(getattr(hidden, "dtype", None)) != "bfloat16"
            or getattr(hidden, "device", None) is not self.device
        ):
            raise ValueError(
                "MTP hidden state must be a BF16 buffer on the target device"
            )
        hidden._check()
        self.device.driver.call(
            "cuMemcpyDtoD_v2", dest.pointer, hidden.pointer, dest.nbytes
        )
        self.device.driver.call("cuStreamSynchronize", None)
        return super().forward(tokens, read_logits=read_logits)

    @property
    def allocated_bytes(self):
        return sum(
            b.nbytes
            for b in (
                *self.owned_weights.values(),
                *self.buffers.values(),
                *(b for state in self.states.values() for b in state),
            )
        )

    def close(self):
        if self.closed:
            return
        self.target.prefills.discard(self)
        # Base teardown only owns the MTP weights. Embedding and head belong to
        # the target and must remain usable after the drafter is closed.
        self.weights = self.owned_weights
        super().close()
        self.owned_weights = {}


class Qwen35MTPPrefill(Qwen35Prefill):
    def __init__(self, mtp, bundle):
        value = json.loads((Path(bundle) / "prefill.json").read_text())
        if (
            value.get("draft") != "qwen35-mtp"
            or value.get("implementation") != fixed_slots_hashes()
        ):
            raise ValueError("MTP prefill source mismatch; rebuild its bundle")
        super().__init__(mtp, bundle)

    def _plan(self):
        super()._plan()
        extra = []

        def call(kind, p, **bindings):
            kernel = self.kernels[identity(kind, p)]
            args = tuple(bindings[a["name"]] for a in kernel.manifest["arguments"])
            extra.append(
                (
                    kernel,
                    bind_call(self.device, kernel, args),
                )
            )

        b, w = self.buffers, self.model.weights
        call(
            "mtp_join",
            dict(r=self.rows, c=2048, eps=self.model.config.epsilon),
            embedding=b["residual"],
            hidden=b["mtp_hidden"],
            embedding_weight=w["mtp.pre_fc_norm_embedding.weight"],
            hidden_weight=w["mtp.pre_fc_norm_hidden.weight"],
            out=b["mtp_join"],
        )
        call(
            "bf16_linear",
            dict(r=self.rows, k=4096, o=2048),
            x=b["mtp_join"],
            w=w["mtp.fc.weight"],
            out=b["mtp_fc"],
        )
        call("mtp_cast", dict(r=self.rows, c=2048), x=b["mtp_fc"], out=b["residual"])
        # Controls and token embedding precede the MTP join; the ordinary
        # one-layer attention/MoE path then consumes the fused BF16 residual.
        self.plan[2:2] = extra

    def forward(self, tokens, hidden, lengths=None, *, read_logits=False):
        self.model._check()
        if self.closed:
            raise RuntimeError("MTP prefill is closed")
        dest = self.buffers["mtp_hidden"]
        if (
            getattr(hidden, "shape", None) != dest.shape
            or str(getattr(hidden, "dtype", None)) != "bfloat16"
            or getattr(hidden, "device", None) is not self.device
        ):
            raise ValueError(
                "MTP prefill needs matching flat BF16 target hidden states"
            )
        hidden._check()
        self.device.driver.call(
            "cuMemcpyDtoD_v2", dest.pointer, hidden.pointer, dest.nbytes
        )
        self.device.driver.call("cuStreamSynchronize", None)
        return super().forward(tokens, lengths, read_logits=read_logits)
