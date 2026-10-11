"""Scratch allocation for the MoE decode geometry; the model owns buffers."""

import numpy as np


def allocate_workspace(model):
    r, c = model.slots, model.config.width

    def alloc(name, shape, dtype="float32", zero=False):
        model.buffers[name] = (
            model.device.from_numpy(np.zeros(shape, dtype=dtype))
            if zero
            else model.device.empty(shape, dtype)
        )

    for name in ("tokens", "positions", "active"):
        alloc(name, (r,), "int32", True)
    for name in ("residual", "normal"):
        alloc(name, (r, c), "bfloat16")
    for name, columns in (
        ("mixed", c),
        ("ffn", c),
        ("qkv", 8192),
        ("z", 4096),
        ("a", 32),
        ("b", 32),
        ("conv", 8192),
        ("attq", 8192),
        ("attk", 512),
        ("attv", 512),
        ("router", 256),
        ("shared_gate", 1),
        ("shared_g", 512),
        ("shared_u", 512),
        ("shared_out", 2048),
    ):
        alloc(name, (r, columns))
    alloc("attq_normal", (r, 16, 256), "bfloat16")
    alloc("att_parts", (r, 2, model.splits, 8, 256))
    alloc("att_stats", (r, 2, model.splits, 8, 2))
    alloc("att_out", (r, 4096), "bfloat16")
    for name in ("gdn_q", "gdn_k", "gdn_v", "gdn_out"):
        alloc(name, (r, 32, 128))
    for name in ("gdn_decay", "gdn_beta"):
        alloc(name, (r, 32))
    alloc("gdn_normal", (r, 4096), "bfloat16")
    alloc("ids", (r, 8), "int32")
    alloc("routes", (r * 8, r), "int32")
    alloc("experts", (r * 8,), "int32")
    alloc("router_weights", (r, 8))
    for name, columns in (
        ("expert_g", 512),
        ("expert_u", 512),
        ("expert_out", 2048),
    ):
        alloc(name, (r, 8, columns))
    alloc("expert_activation", (r, 8, 512), "bfloat16")
    alloc("shared_activation", (r, 512), "bfloat16")
    alloc("logits", (r, model.config.vocab))
