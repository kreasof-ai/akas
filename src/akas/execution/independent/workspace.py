"""Scratch layouts for LFM2 execution profiles; buffers belong to the model."""

import numpy as np
from akas.profiles.independent.parameters import Q16_PREFILL
from akas.profiles.independent.parameters import SUBGROUP_PROFILES


def allocate_workspaces(model, upload):
    model.workspaces = {}
    for r in model.rows:
        c, f, kh, d = (
            model.config.width,
            model.config.ff,
            model.config.kv_heads,
            model.config.head_dim,
        )
        shapes = {
            "tokens": (r, np.int32),
            "hidden": (r * c, np.float32),
            "normal": (r * c, np.float32),
            "mixed": (r * c, np.float32),
            "project": (r * 3 * c, np.float32),
            "convout": (r * c, np.float32),
            "q": (r * c, np.float32),
            "k": (r * kh * d, np.float32),
            "v": (r * kh * d, np.float32),
            "qo": (r * c, np.float32),
            "attn": (r * c, np.float32),
            "gate": (r * f, np.float32),
            "up": (r * f, np.float32),
            "activated": (r * f, np.float32),
            "last": (c, np.float32),
            "final": (c, np.float32),
        }
        model.workspaces[r] = {
            name: upload(np.zeros(count, dtype))
            for name, (count, dtype) in shapes.items()
        }
        if model.provider == "webgpu" or model.cuda_optimized:
            model.workspaces[r]["hidden2"] = upload(np.zeros(r * c, np.float32))
        if model.cuda_optimized and r == 1:
            model.workspaces[r]["parts"] = upload(
                np.zeros(model.config.heads * 16 * 66, np.float32)
            )
        if model.prefill_last and (r == 128 or model.cuda_optimized and r > 1):
            for name, count in (
                ("last_normal", c),
                ("last_activated", f),
                ("last_output", c),
            ):
                model.workspaces[r][name] = upload(np.zeros(count, np.float32))
        if model.prefill_tail and r == 128:
            t = model.prefill_tail
            tail_shapes = {
                "hidden": t * c,
                "hidden2": t * c,
                "normal": t * c,
                "mixed": t * c,
                "project": t * 3 * c,
                "convout": t * c,
                "activated": t * f,
                "last": c,
                "final": c,
                "q": t * c,
                "qo": t * c,
                "attn": t * c,
                "last_normal": c,
                "last_activated": f,
                "last_output": c,
            }
            model.workspaces[r]["tail"] = {
                name: upload(np.zeros(count, np.float32))
                for name, count in tail_shapes.items()
            }
            model.workspaces[r]["tail_control"] = upload(np.zeros(2, np.int32))
        if (
            model.provider == "webgpu"
            and model.webgpu_profile in SUBGROUP_PROFILES
            and r == 1
        ):
            model.workspaces[r]["scores"] = upload(
                np.zeros(model.config.heads * model.capacity, np.float32)
            )
        if (
            model.provider == "webgpu"
            and model.webgpu_profile in ("prefill_q16", "prefill_mixed")
            and r > 1
        ):
            model.workspaces[r]["q16"] = {
                k: (
                    upload(np.zeros(r * k // 2, np.uint32)),
                    upload(np.zeros(r * k // 16, np.float32)),
                    upload(np.zeros(r * k // 16, np.int32)),
                )
                for k in sorted({key[2] for key in Q16_PREFILL if key[1] == r})
            }
