"""Official dense MTP branch sharing the target's embedding and head."""

from dataclasses import replace
from akas.execution.bindings.bf16.resources import BF16Binding as DenseEngine
from akas.memory.layouts import allocate_hybrid_states


class DenseMTP(DenseEngine):
    def __init__(self, target):
        target._check()
        self.target = target
        self.draft = True
        from akas.execution.executor import Executor

        Executor.__init__(
            self, capacity=target.capacity, context=target.context, device=target.device
        )
        self.fused_head = target.fused_head
        self.implementation = target.implementation
        self.checkpoint = target.checkpoint
        self.config = replace(target.config, layers=("full_attention",))
        self.architecture = target.definition(self.config, self.checkpoint)
        self.bundle = target.bundle
        self.batches = {}
        self.owned_weights = self.checkpoint.upload(self.device, branch="mtp")
        self.weights = {}
        for name, buffer in self.owned_weights.items():
            alias = name.replace("mtp.layers.0.", "model.language_model.layers.0.")
            if name == "mtp.norm.weight":
                alias = "model.language_model.norm.weight"
            self.weights[alias] = buffer
        self.weights["model.language_model.embed_tokens.weight"] = target.weights[
            "model.language_model.embed_tokens.weight"
        ]
        c = self.config
        self.states = allocate_hybrid_states(
            self.device, self.config, self.capacity, self.context
        )
        self.hidden = self.device.empty((self.capacity, c.width), "bfloat16")

    def _check(self):
        self.target._check()
        super()._check()

    def _close_resources(self):
        for batch in self.batches.values():
            batch.close()
        for state in self.states.values():
            for buffer in state:
                buffer.release()
        self.hidden.release()
        for weight in self.owned_weights.values():
            weight.release()
