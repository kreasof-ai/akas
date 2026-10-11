"""Qwen dense plan for a fixed execution geometry."""

from __future__ import annotations
from types import SimpleNamespace
from akas.ops.attention import compose as attention
from akas.ops.gdn import compose as gdn
from akas.execution.plan import ExecutionPlan
from akas.execution.plan import KernelPlan
from akas.ops.linear.compose import split_linear
from akas.ops.normalization.compose import rms_norm
from akas.profiles.compact_batch.parameters import schedule

from akas.ops.ffn.compose import bf16_batch_ffn as ffn_bf16_batch_ffn


def build_plan(self):
    c = self.owner.config
    w = self.owner.weights
    b = self.buffers
    r = self.rows
    s = self.slots
    common = dict(slots=s, chunk=self.chunk, pool=self.owner.capacity)
    calls = ExecutionPlan(self.device)
    forward = KernelPlan(calls, self.kernels)
    commit = KernelPlan(self.commit_calls, self.kernels)

    def add(kind, p, *args, to_commit=False, **bindings):
        (commit if to_commit else forward)(kind, p, *args, **bindings)

    def linear(name, x, out, rows=r):
        weight = w[name + ".weight"]
        o, k = weight.shape
        dest = (
            "head"
            if o == c.vocab
            else (
                "gdn"
                if o == c.qkv_width + c.value_width + 2 * c.value_heads
                else (
                    "att"
                    if o == 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim
                    else (
                        "mlp"
                        if o == 2 * c.intermediate
                        else "down"
                        if k == c.intermediate
                        else "out"
                    )
                )
            )
        )
        p = schedule(rows, k, o, c.width)
        split_linear(
            add,
            self.device,
            x,
            weight,
            out,
            rows=rows,
            parameters=p,
            scratch=b.get("projection_" + dest),
            views=self.views,
        )

    def norm(name, x=None):
        rms_norm(
            add,
            b["residual"] if x is None else x,
            w[name + ".weight"],
            b["normal"],
            rows=r,
            width=c.width,
            epsilon=c.epsilon,
            residual=None if x is None else b["residual"],
            input_name="residual",
        )

    add(
        "controls", common, b["position"], b["lengths"], b["flat_position"], b["active"]
    )
    add(
        "embedding",
        dict(r=r, c=c.width, vocab=c.vocab),
        b["tokens"],
        w[self.owner.architecture.embedding],
        b["residual"],
    )
    if self.draft:
        add(
            "gather_hidden",
            dict(common, c=c.width),
            self.owner.hidden,
            b["mapping"],
            b["mode"],
            b["hidden"],
        )
        add(
            "mtp_join",
            dict(r=r, c=c.width, eps=c.epsilon),
            b["residual"],
            b["hidden"],
            w["mtp.pre_fc_norm_embedding.weight"],
            w["mtp.pre_fc_norm_hidden.weight"],
            b["join"],
        )
        linear("mtp.fc", b["join"], b["mixed"])
        add("mtp_cast", dict(r=r, c=c.width), b["mixed"], b["residual"])

    def input_norm(ctx, block):
        ctx.norm(block.input_norm, None if block.index == 0 else ctx.buffers["ffn"])

    def post_norm(ctx, block):
        ctx.norm(block.post_norm, ctx.buffers["mixed"])

    ctx = SimpleNamespace(
        executor=self,
        model=self,
        weights=w,
        buffers=b,
        rows=r,
        slots=s,
        config=c,
        common=common,
        emit=add,
        linear=linear,
        norm=norm,
        input_norm=input_norm,
        post_norm=post_norm,
        mixers={
            "linear_attention": gdn.bf16_batch_gdn,
            "full_attention": attention.bf16_batch_attention,
        },
        feed_forwards={"dense": ffn_bf16_batch_ffn},
    )
    for block in self.owner.architecture.blocks:
        block.compose(ctx)
    norm(self.owner.architecture.final_norm.removesuffix(".weight"), b["ffn"])
    if not self.verify:
        add("last_rows", dict(common, c=c.width), b["normal"], b["lengths"], b["last"])
    if self.draft:
        add(
            "store_hidden",
            dict(common, c=c.width),
            b["last"],
            b["mapping"],
            b["lengths"],
            self.owner.hidden,
        )
    if self.owner.fused_head:
        add(
            "head_linear",
            schedule(self.head_rows, c.width, c.vocab, c.width),
            b["normal"] if self.verify else b["last"],
            w[self.owner.architecture.embedding],
            b["head_maximum"],
            b["head_index"],
        )
        add(
            "head_argmax",
            dict(r=self.head_rows, tiles=(c.vocab + 63) // 64, vocab=c.vocab),
            b["head_maximum"],
            b["head_index"],
            b["head_active"],
            b["predicted"],
        )
    else:
        linear(
            self.owner.architecture.head.removesuffix(".weight"),
            b["normal"] if self.verify else b["last"],
            b["logits"],
            rows=self.head_rows,
        )
        add(
            "argmax",
            dict(r=self.head_rows, vocab=c.vocab),
            b["logits"],
            b["predicted"],
            b["head_position"],
            b["head_active"],
        )
    return calls
