"""Static kernel coverage required by an LFM2 configuration."""

from __future__ import annotations
import hashlib
from importlib.util import find_spec
from pathlib import Path
from akas.models.lfm2.config import Config
from akas.kernels.packed.ggml import identity
from akas.kernels.packed.cuda import CUDA_PROFILES
from akas.execution.profiles.packed import Q16_PREFILL, SUBGROUP_PROFILES, WEBGPU_PROFILES
from akas.execution.profiles.selection import paired_ffn, prefill_tail_rows, projection_tile, webgpu_parameters


def requirements(
    gguf,
    capacity,
    rows=(1, 128),
    *,
    provider="cuda",
    webgpu_profile="portable",
    cuda_profile="default",
    config=None,
):
    cfg = config if config is not None else Config.from_gguf(gguf)
    c = cfg.width
    values = {}
    if webgpu_profile not in WEBGPU_PROFILES:
        raise ValueError("unsupported WebGPU kernel profile")
    if cuda_profile not in CUDA_PROFILES:
        raise ValueError("unsupported CUDA kernel profile")
    optimized = provider == "cuda" and cuda_profile == "optimized"

    def add(kind, **p):
        if provider == "webgpu":
            p = webgpu_parameters(kind, p, webgpu_profile)
        values[identity(kind, p)] = (kind, p)

    for r in rows:
        for info in gguf.tensors.values():
            if len(info.shape) == 2 and not info.name.endswith("conv.weight"):
                o, k = info.shape
                if info.name == "token_embd.weight":
                    add("embedding", r=r, c=c, v=cfg.vocab, type=info.type)
                    add("linear", r=1, k=k, o=o, type=info.type)
                elif info.name == "output.weight":
                    add("linear", r=1, k=k, o=o, type=info.type)
                else:
                    add(
                        "linear", r=r, k=k, o=o, type=info.type, **projection_tile(r, o)
                    )
                    if (
                        provider == "webgpu"
                        and webgpu_profile in ("prefill_q16", "prefill_mixed")
                        and r > 1
                        and any(key[1] == r and key[2] == k for key in Q16_PREFILL)
                    ):
                        add("quantize_q16", r=r, k=k, sg=True)
                    if (
                        (provider == "webgpu" or optimized)
                        and r == 1
                        and info.name.endswith("ffn_down.weight")
                    ):
                        add("linear_add", r=r, k=k, o=o, type=info.type)
        add("rms", r=r, c=c, eps=cfg.epsilon)
        if provider == "webgpu" or optimized:
            if r == 1:
                add("add_rms", r=r, c=c, eps=cfg.epsilon)
            for i in range(len(cfg.layers)):
                gate, up = (
                    gguf.tensors[f"blk.{i}.ffn_{name}.weight"]
                    for name in ("gate", "up")
                )
                if paired_ffn(
                    r, gate, up, "cuda_optimized" if optimized else webgpu_profile
                ):
                    add(
                        "ffn",
                        r=r,
                        k=c,
                        o=cfg.ff,
                        type=gate.type,
                        **projection_tile(r, cfg.ff),
                    )
        add("add", r=r, c=c)
        add("swiglu", r=r, c=cfg.ff)
        add("conv", r=r, c=c)
        for kind in (("qnorm", "kvnorm") if provider == "webgpu" else ("qkv",)):
            add(
                kind,
                r=r,
                h=cfg.heads,
                kh=cfg.kv_heads,
                d=cfg.head_dim,
                cap=capacity,
                eps=cfg.epsilon,
                theta=cfg.theta,
            )
        if provider == "webgpu" and webgpu_profile in SUBGROUP_PROFILES and r == 1:
            add(
                "attention_scores",
                r=r,
                h=cfg.heads,
                kh=cfg.kv_heads,
                d=cfg.head_dim,
                cap=capacity,
            )
        if optimized and r == 1:
            add(
                "attention_partial",
                r=r,
                h=cfg.heads,
                kh=cfg.kv_heads,
                d=cfg.head_dim,
                cap=capacity,
                splits=16,
            )
            if (cfg.heads, cfg.kv_heads, cfg.head_dim) == (32, 8, 64):
                add(
                    "attention_grouped",
                    r=r,
                    h=cfg.heads,
                    kh=cfg.kv_heads,
                    d=cfg.head_dim,
                    cap=capacity,
                    splits=16,
                )
            add("attention_merge", h=cfg.heads, d=cfg.head_dim, splits=16)
        else:
            add(
                "attention",
                r=r,
                h=cfg.heads,
                kh=cfg.kv_heads,
                d=cfg.head_dim,
                cap=capacity,
            )
        add("last", r=r, c=c)
        add("advance", r=r)
    add("rms", r=1, c=c, eps=cfg.epsilon)
    if optimized:
        prefix = f"blk.{len(cfg.layers)-1}."
        gate, up = (
            gguf.tensors[prefix + "ffn_" + name + ".weight"] for name in ("gate", "up")
        )
        if gate.type == up.type:
            add("ffn", r=1, k=c, o=cfg.ff, type=gate.type)
    tail = (
        prefill_tail_rows(cfg, gguf, "cuda_optimized" if optimized else webgpu_profile)
        if (provider == "webgpu" or optimized) and 128 in rows
        else 0
    )
    if tail:
        if optimized:
            add(
                "kvnorm",
                r=128,
                h=cfg.heads,
                kh=cfg.kv_heads,
                d=cfg.head_dim,
                cap=capacity,
                eps=cfg.epsilon,
                theta=cfg.theta,
            )
        add("prefill_tail", r=128, c=c, t=tail)
        for i in range(len(cfg.layers) - 3, len(cfg.layers)):
            prefix = f"blk.{i}."
            names = ["ffn_down"] + (
                ["shortconv.in_proj", "shortconv.out_proj"]
                if cfg.layers[i] == "conv"
                else ["attn_q", "attn_output"]
            )
            for name in names:
                info = gguf.tensors[prefix + name + ".weight"]
                o, k = info.shape
                add(
                    "linear",
                    r=tail,
                    k=k,
                    o=o,
                    type=info.type,
                    **projection_tile(tail, o),
                )
            gate = gguf.tensors[prefix + "ffn_gate.weight"]
            add(
                "ffn",
                r=tail,
                k=c,
                o=cfg.ff,
                type=gate.type,
                **projection_tile(tail, cfg.ff),
            )
        add("rms", r=tail, c=c, eps=cfg.epsilon)
        add("add", r=tail, c=c)
        add("conv", r=tail, c=c)
        add("last", r=tail, c=c)
        add(
            "attention",
            r=tail,
            h=cfg.heads,
            kh=cfg.kv_heads,
            d=cfg.head_dim,
            cap=capacity,
        )
        add(
            "qnorm",
            r=tail,
            h=cfg.heads,
            kh=cfg.kv_heads,
            d=cfg.head_dim,
            cap=capacity,
            eps=cfg.epsilon,
            theta=cfg.theta,
        )
    if provider == "webgpu" or optimized:
        add("argmax", n=cfg.vocab)
    return values


def implementation_hashes(provider="cuda"):
    import tensor.runtime.abi
    import tensor.runtime.signature
    import tensor.providers.cuda
    import tensor.providers.cuda.graph
    import tensor.artifacts.format

    modules = (
        tensor.runtime.abi,
        tensor.runtime.signature,
        tensor.providers.cuda,
        tensor.providers.cuda.graph,
        tensor.artifacts.format,
    )
    names = (
        "akas.execution.packed.model",
        "akas.execution.packed.request",
        "akas.execution.packed.plan",
        "akas.execution.profiles.packed",
        "akas.execution.profiles.selection",
        "akas.generation.greedy",
        "akas.tensor.executor",
        "akas.tensor.graph",
        "akas.models.lfm2.model",
        "akas.models.lfm2.config",
        "akas.models.lfm2.weights",
        "akas.execution.packed.workspace",
        "akas.models.lfm2.config",
        "akas.kernels.packed.ggml",
        "akas.checkpoints.gguf",
        "akas.text.tokenizer",
        "akas.text.formats",
        "akas.text.chat",
        "akas.artifacts.kernels",
        "akas.execution.plan",
        "akas.memory.state",
        "akas.ops.hybrid_state",
        "akas.ops.decoder",
        "akas.ops.attention",
        "akas.ops.gdn",
        "akas.ops.shortconv",
        "akas.ops.ffn",
        "akas.checkpoints.upload",
        "akas.ops.linear",
        "akas.ops.decoder",
        "akas.ops.normalization",
        "akas.ops.quantization",
        "akas.kernels.elementwise",
        __name__,
    )
    if provider == "cuda":
        names += ("akas.kernels.packed.cuda",)
    if provider == "webgpu":
        import tensor.providers.webgpu
        import tensor.artifacts.webgpu

        modules = (
            tensor.runtime.abi,
            tensor.runtime.signature,
            tensor.providers.webgpu,
            tensor.artifacts.webgpu,
            tensor.artifacts.format,
        )
        names += ("akas.kernels.packed.webgpu",)
    result = {
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in modules
    }
    names += (
        "tensor.runtime.buffer", "tensor.runtime.executable", "tensor.runtime.session",
        "tensor.runtime.event", "tensor.runtime.errors", "tensor.runtime.dtypes",
        "tensor.runtime.dlpack",
    )
    if provider == "cuda":
        names += ("tensor.providers.cuda.device", "tensor.providers.cuda.driver")
    else:
        names += (
            "tensor.providers.webgpu.device", "tensor.providers.webgpu.adapter",
            "tensor.providers.webgpu.buffer", "tensor.providers.webgpu.plan",
        )
    for name in names:
        result[name] = hashlib.sha256(
            Path(find_spec(name).origin).read_bytes()
        ).hexdigest()
    if provider == "webgpu":
        native = (
            Path(find_spec("tensor").origin).parent
            / "native"
            / "webgpu_plan.c"
        )
        result["tensor.native.webgpu_plan"] = hashlib.sha256(
            native.read_bytes()
        ).hexdigest()
    return result
