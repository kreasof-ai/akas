"""Qwen decode layer composition and kernel bindings."""

from __future__ import annotations

from types import SimpleNamespace
from akas.ops import attention, gdn, ffn
from akas.execution.plan import ExecutionPlan, KernelPlan
from akas.ops.linear import linear
from akas.ops.normalization import rms_norm


def build_decode_plan(self):
    w, b, r = (self.weights, self.buffers, self.slots)
    calls = ExecutionPlan(self.device, labeled=True)
    add = KernelPlan(calls, self.kernels)

    def fp8(name, x, out):
        linear(
            add,
            x,
            w[name + ".weight"],
            out,
            rows=r,
            scales=w[name + ".weight_scale_inv"],
        )

    def bf16(name, x, out):
        linear(add, x, w[name + ".weight"], out, rows=r)

    def norm(name, x=None, label=None):
        rms_norm(
            add,
            b["residual"] if x is None else x,
            w[name + ".weight"],
            b["normal"],
            rows=r,
            width=2048,
            epsilon=self.config.epsilon,
            residual=None if x is None else b["residual"],
            input_name="residual",
            label=label,
        )

    add(
        "embedding",
        dict(r=r, c=2048, vocab=self.config.vocab),
        b["tokens"],
        w[self.architecture.embedding],
        b["residual"],
    )

    def input_norm(ctx, block):
        ctx.norm(
            block.input_norm,
            None if block.index == 0 else ctx.buffers["ffn"],
            label=(block.index, "input"),
        )

    def post_norm(ctx, block):
        ctx.norm(
            block.post_norm, ctx.buffers["mixed"], label=(block.index, "post_attention")
        )

    ctx = SimpleNamespace(
        executor=self,
        model=self,
        weights=w,
        buffers=b,
        rows=r,
        slots=self.slots,
        emit=add,
        fp8=fp8,
        bf16=bf16,
        norm=norm,
        input_norm=input_norm,
        post_norm=post_norm,
        mixers={
            "linear_attention": gdn.fp8_decode_gdn,
            "full_attention": attention.fp8_decode_attention,
        },
        feed_forwards={"moe": ffn.fp8_decode_moe},
    )
    for block in self.architecture.blocks:
        block.compose(ctx)
    norm(
        self.architecture.final_norm.removesuffix(".weight"),
        b["ffn"],
        label=(40, "final"),
    )
    bf16(self.architecture.head.removesuffix(".weight"), b["normal"], b["logits"])
    add(
        "argmax",
        dict(r=r, vocab=self.config.vocab),
        b["logits"],
        b["tokens"],
        b["positions"],
        b["active"],
    )
    return calls
