"""Qwen prefill layer composition and kernel bindings."""

from types import SimpleNamespace
from akas.ops import attention, gdn, ffn
from akas.execution.plan import ExecutionPlan, KernelPlan
from akas.ops.linear import linear
from akas.ops.normalization import rms_norm
from akas.ops.quantization import ActivationCache


def build_prefill_plan(self):
    model = self.model
    b = self.buffers
    w = model.weights
    d = self.device
    r = self.rows
    s = model.slots
    self.plan = ExecutionPlan(d)
    activation_cache = ActivationCache(b, self.quantized, rows=r)
    call = KernelPlan(self.plan, self.kernels, on_write=activation_cache.invalidate)

    def small(kind, **bindings):
        call(kind, dict(slots=s, chunk=self.chunk), **bindings)

    def quantize(name):
        return activation_cache.get(name, call)

    def fp8(name, x, out):
        bits, scales = quantize(x)
        linear(
            call,
            bits,
            w[name + ".weight"],
            b[out],
            rows=r,
            scales=w[name + ".weight_scale_inv"],
            activation_scales=scales,
        )

    def bf16(name, x, out):
        linear(call, b[x], w[name + ".weight"], b[out], rows=r)

    def norm(name, x=None):
        rms_norm(
            call,
            b["residual"] if not x else b[x],
            w[name + ".weight"],
            b["normal"],
            rows=r,
            width=2048,
            epsilon=model.config.epsilon,
            residual=None if not x else b["residual"],
            input_name="residual",
        )

    def experts(name, x, out, o, k, routed):
        xx, scale = quantize(x)
        call(
            "fp8_experts",
            dict(r=r, experts=256, top=8, o=o, k=k, routed_input=routed),
            x=xx,
            activation_scales=scale,
            w=w[name + ".weight"],
            scales=w[name + ".weight_scale_inv"],
            counts=b["counts"],
            routes=b["routes"],
            out=b[out],
        )

    small(
        "controls",
        position=model.buffers["positions"],
        lengths=b["lengths"],
        flat_position=b["flat_positions"],
        active=b["flat_active"],
    )
    call(
        "embedding",
        dict(r=r, c=2048, vocab=model.config.vocab),
        ids=b["tokens"],
        w=w[model.architecture.embedding],
        out=b["residual"],
    )

    def input_norm(ctx, block):
        ctx.norm(block.input_norm, None if block.index == 0 else "ffn")

    def post_norm(ctx, block):
        ctx.norm(block.post_norm, "mixed")

    ctx = SimpleNamespace(
        executor=self,
        model=model,
        weights=w,
        buffers=b,
        rows=r,
        slots=s,
        emit=call,
        small=small,
        fp8=fp8,
        bf16=bf16,
        norm=norm,
        experts=experts,
        quantize=quantize,
        input_norm=input_norm,
        post_norm=post_norm,
        mixers={
            "linear_attention": gdn.fp8_prefill_gdn,
            "full_attention": attention.fp8_prefill_attention,
        },
        feed_forwards={"moe": ffn.fp8_prefill_moe},
    )
    for block in model.architecture.blocks:
        block.compose(ctx)
    norm(model.architecture.final_norm.removesuffix(".weight"), "ffn")
    call(
        "last_rows",
        dict(slots=s, chunk=self.chunk, width=2048),
        x=b["normal"],
        lengths=b["lengths"],
        out=model.buffers["normal"],
    )
    for kernel, bound, label in model.plan[-2:]:
        self.plan.append((kernel, bound))
    small(
        "advance",
        position=model.buffers["positions"],
        lengths=b["lengths"],
        active=model.buffers["active"],
    )
