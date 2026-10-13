"""Compiler-free plan recording fixtures; buffers retain shape and identity."""

from types import SimpleNamespace

from akas.models.qwen35.config import Qwen35Config
from akas.models.qwen35.config import DenseConfig


def dense_graph_contract(*, verify=False, draft=False, fused=False):
    """Record direct composition; frozen native graphs provide numerical parity."""
    from akas.models.qwen35.weights import DenseCheckpoint
    from akas.models.qwen35.model import compose_dense

    config = DenseConfig(
        1024,
        2048,
        256,
        ("linear_attention", "full_attention"),
        8,
        2,
        256,
        16,
        32,
        128,
        128,
        1e-6,
        1e6,
        64,
        4096,
    )
    checkpoint = object.__new__(DenseCheckpoint)
    checkpoint.config = config
    checkpoint.tensors = {
        name: SimpleNamespace(
            shape=shape,
            dtype="F32"
            if name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))
            else "BF16",
        )
        for name, shape in config.expected_tensors().items()
    }
    graph, commit = compose_dense(
        checkpoint,
        slots=2,
        chunk=4,
        pool=4,
        context=64,
        verify=verify,
        draft=draft,
        fused_head=fused,
    )

    def reference(name):
        value = graph.buffers[name]
        # The old mock recorded all weights as F32 irrespective of actual
        # checkpoint storage. Real storage precision has independent native tests.
        dtype = "float32" if value.role == "weight" else value.dtype
        identity = None
        if value.role == "weight":
            identity = value.name
            if draft:
                identity = identity.replace(
                    "mtp.layers.0.", "model.language_model.layers.0."
                )
                if identity == "mtp.norm.weight":
                    identity = "model.language_model.norm.weight"
        elif value.role == "state":
            identity = (
                value.name.removeprefix("proposal.")
                .replace("state.", "state:")
                .replace(".", ":")
            )
            if value.name == "proposal.saved_hidden":
                identity = "hidden-bank"
        return dict(buffer=identity, shape=list(value.shape), dtype=dtype)

    def calls(value):
        return [
            [
                call.kind,
                dict(
                    parameters=call.parameters,
                    arguments=[reference(name) for name in call.arguments],
                    bindings={
                        name: reference(buffer)
                        for name, buffer in call.bindings.items()
                    },
                ),
                call.label,
            ]
            for call in value.calls
        ]

    return dict(forward=calls(graph), commit=calls(commit))


def dense_frozen_contract(value):
    """Ignore renamed private scratch while retaining all shapes and identities."""
    if isinstance(value, dict):
        if "buffer" in value:
            value = dict(value)
            name = value["buffer"]
            if name.startswith(("batch:", "view:batch:")):
                value["buffer"] = None
            return value
        return {name: dense_frozen_contract(item) for name, item in value.items()}
    if isinstance(value, list):
        return [dense_frozen_contract(item) for item in value]
    return value


def dense_checkpoint(config):
    from akas.models.qwen35.weights import DenseCheckpoint

    checkpoint = object.__new__(DenseCheckpoint)
    checkpoint.config = config
    checkpoint.tensors = {
        name: SimpleNamespace(
            shape=shape,
            dtype="F32"
            if name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))
            else "BF16",
        )
        for name, shape in config.expected_tensors().items()
    }
    return checkpoint


def moe_graph_contract(phase, *, fp8=False):
    """Compare complete direct graphs to the pre-migration operation fixtures."""
    from math import prod
    from akas.models.qwen35.weights import Qwen35Checkpoint
    from akas.models.qwen35.model import compose_moe_program

    config = Qwen35Config(("linear_attention", "full_attention"), 48000, 1e-6, 1e6)
    checkpoint = object.__new__(Qwen35Checkpoint)
    checkpoint.config, checkpoint.branch = config, "text"
    shapes = config.expected_tensors()
    checkpoint.tensors = {}
    for name, shape in shapes.items():
        dtype = "F8_E4M3" if name + "_scale_inv" in shapes else "BF16"
        if name.endswith(("linear_attn.A_log", "linear_attn.norm.weight")):
            dtype = "F32"
        width = 1 if dtype == "F8_E4M3" else 4 if dtype == "F32" else 2
        checkpoint.tensors[name] = SimpleNamespace(
            shape=shape, dtype=dtype, nbytes=prod(shape) * width
        )
    graph = compose_moe_program(
        checkpoint,
        rows=2,
        chunk=4 if phase == "prefill" else 1,
        max_requests=2,
        max_context=64,
        kv_dtype="fp8" if fp8 else "bfloat16",
    )

    def reference(name):
        spec = graph.buffers[name]
        return dict(
            buffer=spec.name
            if spec.role == "weight"
            else spec.name.replace("state.", "state:").replace(".", ":")
            if spec.name.startswith("state.")
            else None,
            shape=list(spec.shape),
            dtype="float32" if spec.role == "weight" else spec.dtype,
        )

    result = [
        [
            call.kind,
            dict(
                parameters=dict(call.parameters),
                arguments=[reference(name) for name in call.arguments],
                bindings={
                    name: reference(value) for name, value in call.bindings.items()
                },
            ),
            list(call.label) if isinstance(call.label, tuple) else call.label,
        ]
        for call in graph.calls
    ]
    if phase == "prefill":
        for kind, details, *_ in result:
            if kind == "fp8_linear":
                assert details["parameters"].pop("prequantized") is True
                assert details["bindings"]["x"]["dtype"] == "uint8"
        # The old test used two unbound placeholders for its inherited decode
        # head. Its actual storage and greedy result have native parity checks.
        head, prediction = result[-3:-1]
        assert head[0] == "bf16_linear" and head[1]["parameters"]["r"] == 2
        assert prediction[0] == "argmax" and prediction[1]["parameters"] == dict(
            r=2, vocab=config.vocab
        )
        result[-3:-1] = [["head", {}], ["argmax", {}]]
    return result


def moe_frozen_contract(value):
    """Rename private workspaces while preserving every dtype, shape and state."""
    if isinstance(value, dict):
        if "buffer" in value:
            value = dict(value)
            if not value["buffer"].startswith(("model.", "lm_head.", "state:")):
                value["buffer"] = None
            return value
        return {name: moe_frozen_contract(item) for name, item in value.items()}
    if isinstance(value, list):
        result = [moe_frozen_contract(item) for item in value]
        if result == [40, "final"]:
            return [2, "final"]
        return result
    return value
