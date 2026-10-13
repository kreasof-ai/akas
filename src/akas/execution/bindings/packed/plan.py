"""LFM2 layer composition; binds operations without owning model or request state."""

from __future__ import annotations
from types import SimpleNamespace
from akas.ops.attention import compose as attention
from akas.ops.convolution import compose as shortconv
from akas.execution.plan import ExecutionPlan
from akas.execution.plan import KernelPlan
from akas.ops.linear.compose import packed_linear
from akas.ops.normalization.operation import rms_norm
from akas.artifacts.schema import identity
from akas.ops.schedules.packed_selection import projection_tile
from akas.ops.schedules.packed_selection import webgpu_parameters

from akas.ops.activation.operation import packed_ffn as ffn_packed_ffn


def build_plan(self, r, *, grouped=False):
    ctx = SimpleNamespace(request=self, rows=r, grouped=grouped)
    ctx.config = ctx.request.config
    ctx.width = ctx.config.width
    ctx.buffers = ctx.request.workspaces[ctx.rows]
    ctx.plan = ExecutionPlan(ctx.request.device)
    ctx.final_hidden = None
    ctx.scheduled_rows = ctx.rows
    ctx.control = ctx.request.control
    ctx.last_only = ctx.request.prefill_last and (
        ctx.rows == 128 or (ctx.request.cuda_optimized and ctx.rows > 1)
    )
    ctx.hidden = ctx.buffers["hidden"]
    ctx.other = ctx.buffers.get("hidden2", ctx.hidden)

    def parameters(kind, p):
        if ctx.request.provider == "webgpu":
            return webgpu_parameters(kind, p, ctx.request.webgpu_profile)
        return p

    ctx.emit = KernelPlan(
        ctx.plan, ctx.request.kernels, identity=identity, parameters=parameters
    )

    def inputs(kind, p, x):
        selected = (
            webgpu_parameters(kind, p, ctx.request.webgpu_profile)
            if ctx.request.provider == "webgpu"
            else p
        )
        if not selected.get("q16"):
            return (x,)
        buffers = ctx.buffers["q16"][p["k"]]
        ctx.emit("quantize_q16", dict(r=p["r"], k=p["k"], sg=True), x, *buffers)
        return buffers

    ctx.inputs = inputs

    def linear(name, x, out, rows=None):
        count = ctx.rows if rows is None else rows
        info = ctx.request.gguf.tensors[name]
        packed_linear(
            ctx.emit,
            lambda kind, p: ctx.inputs(kind, p, x),
            ctx.request.weights[name],
            out,
            rows=count,
            shape=info.shape,
            quantization=info.type,
            tile=projection_tile(count, info.shape[0]),
        )

    ctx.linear = linear

    def rms(name, x, out, rows=None):
        rms_norm(
            ctx.emit,
            x,
            ctx.request.weights[name],
            out,
            rows=ctx.rows if rows is None else rows,
            width=ctx.width,
            epsilon=ctx.config.epsilon,
        )

    ctx.norm = rms
    ctx.emit(
        "embedding",
        dict(
            r=ctx.rows,
            c=ctx.width,
            v=ctx.config.vocab,
            type=ctx.request.gguf.tensors[ctx.request.architecture.embedding].type,
        ),
        ctx.buffers["tokens"],
        ctx.request.weights[ctx.request.architecture.embedding],
        ctx.hidden,
    )

    def input_norm(ctx, block):
        ctx.norm(block.input_norm, ctx.hidden, ctx.buffers["normal"])

    ctx.input_norm = input_norm
    ctx.post_norm = lambda context, block: None
    ctx.mixers = {
        "conv": shortconv.packed_shortconv,
        "attention": attention.packed_attention,
    }
    ctx.feed_forwards = {"packed": ffn_packed_ffn}
    for block in ctx.request.architecture.blocks:
        if block.compose(ctx):
            break
    if ctx.final_hidden is None:
        if (
            ctx.request.provider == "webgpu" or ctx.request.cuda_optimized
        ) and ctx.rows == 1:
            ctx.final_hidden = ctx.hidden
        else:
            ctx.emit(
                "last",
                dict(r=ctx.rows, c=ctx.width),
                ctx.hidden,
                ctx.buffers["last"],
                ctx.control,
            )
            ctx.final_hidden = ctx.buffers["last"]
    ctx.norm(
        ctx.request.architecture.final_norm,
        ctx.final_hidden,
        ctx.buffers["final"],
        rows=1,
    )
    ctx.linear(
        ctx.request.architecture.head, ctx.buffers["final"], ctx.request.logits, rows=1
    )
    ctx.emit("advance", dict(r=ctx.scheduled_rows), ctx.request.control)
    return ctx.plan
