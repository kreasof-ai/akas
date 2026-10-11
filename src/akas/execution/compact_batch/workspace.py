"""Qwen dense workspace for a fixed execution geometry."""

from __future__ import annotations
from akas.profiles.compact_batch.parameters import schedule


def allocate_workspace(self):
    c = self.owner.config
    r = self.rows
    s = self.slots

    def alloc(name, shape, dtype="float32"):
        self.buffers[name] = self.device.empty(shape, dtype)

    for name, shape in (
        ("tokens", (r,)),
        ("mapping", (s,)),
        ("position", (s,)),
        ("lengths", (s,)),
        ("flat_position", (r,)),
        ("active", (r,)),
        ("predicted", (self.head_rows,)),
        ("head_position", (self.head_rows,)),
        ("head_active", (self.head_rows,)),
    ):
        alloc(name, shape, "int32")
    for name in ("residual", "normal"):
        alloc(name, (r, c.width), "bfloat16")
    for name in ("mixed", "ffn"):
        alloc(name, (r, c.width))
    alloc("gdn_input", (r, c.qkv_width + c.value_width + 2 * c.value_heads))
    alloc("conv", (r, c.qkv_width))
    for name in ("q", "k", "v", "gdn_out"):
        alloc(name, (r, c.value_heads, c.key_dim))
    for name in ("g", "beta"):
        alloc(name, (r, c.value_heads))
    alloc("gdn_normal", (r, c.value_width), "bfloat16")
    alloc("att_input", (r, 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim))
    alloc("att_q", (r, c.heads, c.head_dim), "bfloat16")
    alloc("att_out", (r, c.heads * c.head_dim), "bfloat16")
    alloc("mlp_input", (r, 2 * c.intermediate))
    alloc("activation", (r, c.intermediate), "bfloat16")
    alloc("last", (s, c.width), "bfloat16")
    if self.owner.fused_head:
        tiles = (c.vocab + 63) // 64
        alloc("head_maximum", (self.head_rows, tiles))
        alloc("head_index", (self.head_rows, tiles), "int32")
    else:
        alloc("logits", (self.head_rows, c.vocab))
    for kind, rows, k, o in (
        ("gdn", r, c.width, c.qkv_width + c.value_width + 2 * c.value_heads),
        ("att", r, c.width, 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim),
        ("mlp", r, c.width, 2 * c.intermediate),
        ("out", r, c.value_width, c.width),
        ("down", r, c.intermediate, c.width),
        ("head", self.head_rows, c.width, c.vocab),
    ):
        parts = schedule(rows, k, o, c.width)["parts"]
        if parts > 1:
            alloc("projection_" + kind, (rows, parts, o))
    if self.verify:
        for layer, kind in enumerate(c.layers):
            if kind != "linear_attention":
                continue
            alloc(
                f"{layer}_gdn_input",
                (r, c.qkv_width + c.value_width + 2 * c.value_heads),
            )
            for name in ("q", "k", "v"):
                alloc(f"{layer}_{name}", (r, c.value_heads, c.key_dim))
            for name in ("g", "beta"):
                alloc(f"{layer}_{name}", (r, c.value_heads))
    if self.draft:
        alloc("hidden", (r, c.width), "bfloat16")
        alloc("join", (r, 2 * c.width), "bfloat16")
        alloc("mode", (1,), "int32")
