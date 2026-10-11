"""Logical Qwen kernel requirements shared by producers and executors."""

import hashlib
from pathlib import Path
from akas.artifacts.kernels import backbone_hashes
from akas.artifacts.kernels import identity


def requirements(config, slots, capacity, *, splits=16, kv_dtype="bfloat16"):
    if kv_dtype not in ("bfloat16", "fp8"):
        raise ValueError("unsupported KV cache precision")
    kv = dict(kv_dtype="fp8") if kv_dtype == "fp8" else {}
    records = {}

    def add(kind, **p):
        records[identity(kind, p)] = (kind, p)

    r = slots
    for o, k in ((8192, 2048), (4096, 2048), (2048, 4096), (512, 2048), (2048, 512)):
        add("fp8_linear", r=r, o=o, k=k)
    for o in (1, 32, 256, config.vocab):
        add("bf16_linear", r=r, k=2048, o=o)
    add("embedding", r=r, c=2048, vocab=config.vocab)
    for kind in ("rms", "add_rms"):
        add(kind, r=r, c=2048, eps=config.epsilon)
    add("router", r=r, experts=256, top=8)
    add("moe_groups", r=r, top=8)
    for o, k, routed in ((512, 2048, False), (2048, 512, True)):
        add("fp8_experts", r=r, experts=256, top=8, o=o, k=k, routed_input=routed)
    add("swiglu", r=r, c=512)
    add("swiglu_experts", r=r, top=8, c=512)
    add("moe_combine", r=r, top=8, c=2048)
    add("gdn_conv", r=r)
    add("gdn_prepare", r=r)
    add("gdn_recurrent", slots=r, heads=32, key=128, value=128)
    add("gdn_norm", r=r, eps=config.epsilon)
    for kind in ("attention_q", "attention_kv"):
        add(
            kind,
            r=r,
            eps=config.epsilon,
            capacity=capacity,
            theta=config.rope_theta,
            **(kv if kind == "attention_kv" else {}),
        )
    add("attention_partial", r=r, capacity=capacity, splits=splits, **kv)
    add("attention_merge", r=r, splits=splits)
    add("argmax", r=r, vocab=config.vocab)
    return records


def implementation_hashes():
    import akas.ops.hybrid_state as common_state
    import akas.execution.plan as common_plan
    import tensor.providers.cuda
    import tensor.providers.cuda_graph
    import tensor.runtime.abi
    import akas.artifacts.kernels as common_artifacts
    import akas.checkpoints.safetensors as common_checkpoint
    from akas.artifacts import hybrid as artifacts
    from akas.models.qwen35 import weights as checkpoint
    import akas.kernels.hybrid.hybrid as decode
    from akas.kernels import elementwise
    import akas.kernels.attention.fp8_kv as fp8_kv
    from akas.execution.optimizations import decode_projections

    modules = (
        common_plan,
        common_state,
        common_artifacts,
        common_checkpoint,
        artifacts,
        checkpoint,
        decode,
        elementwise,
        decode_projections,
        fp8_kv,
        tensor.providers.cuda,
        tensor.providers.cuda_graph,
        tensor.runtime.abi,
    )
    result = {
        m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in modules
    }
    result[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from importlib.util import find_spec

    for name in (
        "akas.execution.hybrid.model",
        "akas.execution.hybrid.workspace",
        "akas.models.qwen35.model",
        "akas.models.qwen35.config",
        "akas.models.qwen35.weights",
        "akas.memory.estimates",
        "akas.execution.hybrid.decode_plan",
        "akas.execution.hybrid.prefill_plan",
    ):
        result[name] = hashlib.sha256(
            Path(find_spec(name).origin).read_bytes()
        ).hexdigest()
    return result | backbone_hashes()
