"""Compiler-free plan recording fixtures; buffers retain shape and identity."""

from types import SimpleNamespace

from akas.models.qwen35.config import Qwen35Config
from akas.models.qwen35.config import DenseConfig
from akas.models.qwen35.model import architecture
from akas.memory.layouts import allocate_hybrid_states


class Buffer:
    def __init__(self, name, shape, dtype="float32"):
        self.name, self.shape, self.dtype = name, tuple(shape), dtype
        self.pointer = 1

    def view(self, shape):
        return Buffer("view:" + self.name, shape, self.dtype)


class Device:
    def empty(self, shape, dtype="float32"):
        return Buffer("scratch", shape, dtype)

    def from_numpy(self, array):
        return Buffer("scratch", array.shape, str(array.dtype))


class Weights(dict):
    def __init__(self, config):
        self.config = config
        self.shapes = config.expected_tensors()

    def __missing__(self, name):
        c = self.config
        shape = self.shapes.get(name)
        if shape is None:
            if name.endswith("linear_attn.input.weight"):
                shape = (c.qkv_width + c.value_width + 2 * c.value_heads, c.width)
            elif name.endswith("self_attn.input.weight"):
                shape = (
                    2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim,
                    c.width,
                )
            elif name.endswith("mlp.input.weight"):
                shape = (2 * c.intermediate, c.width)
            elif name == "mtp.fc.weight":
                shape = (c.width, 2 * c.width)
            elif ".experts." in name:
                down = "down_proj" in name
                shape = (
                    c.experts,
                    c.width if down else c.expert_width,
                    c.expert_width if down else c.width,
                )
                if name.endswith("scale_inv"):
                    shape = (shape[0], shape[1] // 128, shape[2] // 128)
            else:
                raise KeyError(name)
        value = self[name] = Buffer(name, shape)
        return value


class Plan(list):
    def __init__(self, device=None, calls=(), *, labeled=False):
        super().__init__(calls)
        self.device = device


class Emitter:
    def __init__(self, plan, kernels, *, identity=None, parameters=None, on_write=None):
        self.plan, self.parameters, self.on_write = plan, parameters, on_write

    def __call__(self, kind, parameters, *args, label=None, **bindings):
        if self.parameters:
            parameters = self.parameters(kind, parameters)
        self.plan.append(
            (
                kind,
                dict(parameters=parameters, arguments=list(args), bindings=bindings),
                label,
            )
        )
        if self.on_write and "out" in bindings:
            self.on_write(bindings["out"])


def label(buffers, prefix):
    for name, buffer in buffers.items():
        buffer.name = prefix + name


def normalize(value):
    if isinstance(value, Buffer):
        return dict(buffer=value.name, shape=list(value.shape), dtype=value.dtype)
    if isinstance(value, dict):
        return {name: normalize(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def _record(module, phase, *, fp8=False, verify=False, draft=False, fused=False):
    if phase in ("decode", "prefill"):
        config = Qwen35Config(("linear_attention", "full_attention"), 48000, 1e-6, 1e6)
        model = SimpleNamespace(
            config=config,
            weights=Weights(config),
            buffers={},
            kernels={},
            slots=2,
            splits=16,
            context=64,
            kv_dtype="fp8" if fp8 else "bfloat16",
            device=Device(),
        )
        model.architecture = architecture(config, None)
        from akas.execution.bindings.block_fp8.workspace import allocate_workspace

        allocate_workspace(model)
        label(model.buffers, "decode:")
        model.states = allocate_hybrid_states(
            model.device, config, model.slots, model.context, kv_dtype=model.kv_dtype
        )
        for index, states in model.states.items():
            for offset, state in enumerate(states):
                state.name = f"state:{index}:{offset}"
        if phase == "decode":
            return normalize(module.build_decode_plan(model))
        from akas.execution.bindings.block_fp8.prefill import Qwen35Prefill

        model.plan = [("head", {}, None), ("argmax", {}, None)]
        executor = SimpleNamespace(
            model=model, device=model.device, rows=8, chunk=4, buffers={}, kernels={}
        )
        Qwen35Prefill._allocate(executor)
        label(executor.buffers, "prefill:")
        module.build_prefill_plan(executor)
        return normalize(executor.plan)
    raise ValueError("use dense_graph_contract for dense composition")


def record(module, phase, **options):
    from unittest.mock import patch

    with (
        patch.object(module, "ExecutionPlan", Plan),
        patch.object(module, "KernelPlan", Emitter),
    ):
        return _record(module, phase, **options)


def dense_graph_contract(*, verify=False, draft=False, fused=False):
    """Record direct composition; frozen native graphs provide numerical parity."""
    from akas.models.qwen35.weights import DenseCheckpoint
    from akas.models.qwen35.model import compose_dense

    config = DenseConfig(1024,2048,256,("linear_attention","full_attention"),
                         8,2,256,16,32,128,128,1e-6,1e6,64,4096)
    checkpoint = object.__new__(DenseCheckpoint)
    checkpoint.config = config
    checkpoint.tensors = {name: SimpleNamespace(shape=shape,
        dtype="F32" if name.endswith(("linear_attn.A_log","linear_attn.norm.weight")) else "BF16")
        for name,shape in config.expected_tensors().items()}
    graph,commit = compose_dense(checkpoint,slots=2,chunk=4,pool=4,context=64,
                                 verify=verify,draft=draft,fused_head=fused)
    def reference(name):
        value = graph.buffers[name]
        # The old mock recorded all weights as F32 irrespective of actual
        # checkpoint storage. Real storage precision has independent native tests.
        dtype = "float32" if value.role == "weight" else value.dtype
        identity = None
        if value.role == "weight":
            identity = value.name
            if draft:
                identity = identity.replace("mtp.layers.0.", "model.language_model.layers.0.")
                if identity == "mtp.norm.weight": identity = "model.language_model.norm.weight"
        elif value.role == "state":
            identity = value.name.removeprefix("proposal.").replace("state.","state:").replace(".",":")
            if value.name == "proposal.saved_hidden": identity = "hidden-bank"
        return dict(buffer=identity,shape=list(value.shape),dtype=dtype)
    def calls(value):
        return [[call.kind,dict(parameters=call.parameters,
                               arguments=[reference(name) for name in call.arguments],
                               bindings={name:reference(buffer) for name,buffer in call.bindings.items()}),call.label]
                for call in value.calls]
    return dict(forward=calls(graph),commit=calls(commit))


def dense_frozen_contract(value):
    """Ignore renamed private scratch while retaining all shapes and identities."""
    if isinstance(value,dict):
        if "buffer" in value:
            value = dict(value)
            name = value["buffer"]
            if name.startswith(("batch:", "view:batch:")):
                value["buffer"] = None
            return value
        return {name:dense_frozen_contract(item) for name,item in value.items()}
    if isinstance(value,list):
        return [dense_frozen_contract(item) for item in value]
    return value


def dense_checkpoint(config):
    from akas.models.qwen35.weights import DenseCheckpoint
    checkpoint = object.__new__(DenseCheckpoint)
    checkpoint.config = config
    checkpoint.tensors = {name: SimpleNamespace(shape=shape,
        dtype="F32" if name.endswith(("linear_attn.A_log","linear_attn.norm.weight")) else "BF16")
        for name,shape in config.expected_tensors().items()}
    return checkpoint
