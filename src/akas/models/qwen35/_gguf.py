"""Qwen GGUF tensor mapping; format arithmetic remains in quantization."""

from functools import cached_property
from math import prod

from akas.checkpoints.gguf import GGUF, GGUFError
from akas.models.qwen35.config import Qwen35DenseConfig, Qwen35MoEConfig


class Checkpoint:
    """A text checkpoint retaining original compressed projection/expert bytes."""

    storage_format = "ggml"

    def __init__(self, path, *, moe, branch="text"):
        if branch not in ("text", "mtp"):
            raise GGUFError("Qwen GGUF branch must be text or mtp")
        self.gguf = GGUF(path)
        self.config = (Qwen35MoEConfig if moe else Qwen35DenseConfig).from_gguf(
            self.gguf
        )
        self.branch = branch
        self.groups, self.auxiliary, self.orders = {}, {}, {}
        self.mtp_groups, self.mtp_used = {}, set()
        self._declare(moe)
        self.tensors = {name: self.gguf.tensors[name] for name in self.used}

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_gguf(self.gguf.metadata)

    def _declare(self, moe):
        c = self.config
        self.used = set()
        self.group_shapes = {}
        self.physical_shapes = {}

        def tensor(
            canonical,
            source,
            shape,
            *,
            auxiliary=False,
            offset_norm=False,
            order=None,
            column_order=None,
            mtp=False,
        ):
            if source not in self.gguf.tensors:
                raise GGUFError(f"missing Qwen GGUF tensor: {source}")
            info = self.gguf.tensors[source]
            allowed = {tuple(shape)}
            if len(shape) == 2 and shape[0] == 1:
                allowed.add((shape[1],))
            if auxiliary:
                allowed.add(tuple(dim for dim in shape if dim != 1))
            if info.shape not in allowed:
                raise GGUFError(f"{source}: expected {shape}, found {info.shape}")
            target_used = self.mtp_used if mtp else self.used
            target_groups = self.mtp_groups if mtp else self.groups
            target_used.add(source)
            self.physical_shapes[source] = tuple(shape)
            target_groups[canonical] = (source,)
            self.group_shapes[canonical] = tuple(shape)
            self.orders[source] = (order, column_order)
            if auxiliary:
                if info.type not in (0, 1):
                    raise GGUFError(
                        f"{source}: auxiliary weights require F32/F16 storage"
                    )
                dtype = (
                    "float32"
                    if canonical.endswith((".A_log", ".linear_attn.norm.weight"))
                    else "bfloat16"
                )
                self.auxiliary[source] = (
                    tuple(shape),
                    dtype,
                    offset_norm,
                    canonical.endswith(".A_log"),
                )

        def join(canonical, names, *, mtp=False):
            target_groups = self.mtp_groups if mtp else self.groups
            sources = tuple(target_groups[name][0] for name in names)
            depth = self.group_shapes[names[0]][-1]
            target_groups[canonical] = sources
            self.group_shapes[canonical] = (
                sum(self.group_shapes[name][0] for name in names),
                depth,
            )

        embedding = "model.language_model.embed_tokens.weight"
        tensor(embedding, "token_embd.weight", (c.vocab, c.width))
        tensor(
            "model.language_model.norm.weight",
            "output_norm.weight",
            (c.width,),
            auxiliary=True,
            offset_norm=True,
        )
        self.head_name = (
            "lm_head.weight" if "output.weight" in self.gguf.tensors else embedding
        )
        if self.head_name != embedding:
            tensor(self.head_name, "output.weight", (c.vocab, c.width))
        ratio = c.value_heads // c.key_heads

        def order(width, prefix=0):
            return (prefix, c.key_heads, ratio, width) if ratio > 1 else None

        qkv = c.qkv_width
        for index, kind in enumerate(c.layers):
            hf, gg = f"model.language_model.layers.{index}.", f"blk.{index}."
            tensor(
                hf + "input_layernorm.weight",
                gg + "attn_norm.weight",
                (c.width,),
                auxiliary=True,
                offset_norm=True,
            )
            tensor(
                hf + "post_attention_layernorm.weight",
                gg + "post_attention_norm.weight",
                (c.width,),
                auxiliary=True,
                offset_norm=True,
            )
            if kind == "linear_attention":
                p = hf + "linear_attn."
                for target, source, shape, layout in (
                    (
                        "in_proj_qkv.weight",
                        "attn_qkv.weight",
                        (qkv, c.width),
                        order(c.value_dim, 2 * c.key_heads * c.key_dim),
                    ),
                    (
                        "in_proj_z.weight",
                        "attn_gate.weight",
                        (c.value_width, c.width),
                        order(c.value_dim),
                    ),
                    (
                        "in_proj_a.weight",
                        "ssm_alpha.weight",
                        (c.value_heads, c.width),
                        order(1),
                    ),
                    (
                        "in_proj_b.weight",
                        "ssm_beta.weight",
                        (c.value_heads, c.width),
                        order(1),
                    ),
                ):
                    tensor(p + target, gg + source, shape, order=layout)
                join(
                    p + "input.weight",
                    tuple(
                        p + "in_proj_" + part + ".weight"
                        for part in ("qkv", "z", "a", "b")
                    ),
                )
                tensor(
                    p + "out_proj.weight",
                    gg + "ssm_out.weight",
                    (c.width, c.value_width),
                    column_order=order(c.value_dim),
                )
                for target, source, shape, layout in (
                    ("A_log", "ssm_a", (c.value_heads,), order(1)),
                    ("dt_bias", "ssm_dt.bias", (c.value_heads,), order(1)),
                    (
                        "conv1d.weight",
                        "ssm_conv1d.weight",
                        (qkv, 1, 4),
                        order(c.value_dim, 2 * c.key_heads * c.key_dim),
                    ),
                    ("norm.weight", "ssm_norm.weight", (c.value_dim,), None),
                ):
                    tensor(p + target, gg + source, shape, auxiliary=True, order=layout)
            else:
                p = hf + "self_attn."
                for target, source, shape in (
                    (
                        "q_proj.weight",
                        "attn_q.weight",
                        (2 * c.heads * c.head_dim, c.width),
                    ),
                    (
                        "k_proj.weight",
                        "attn_k.weight",
                        (c.kv_heads * c.head_dim, c.width),
                    ),
                    (
                        "v_proj.weight",
                        "attn_v.weight",
                        (c.kv_heads * c.head_dim, c.width),
                    ),
                    (
                        "o_proj.weight",
                        "attn_output.weight",
                        (c.width, c.heads * c.head_dim),
                    ),
                ):
                    tensor(p + target, gg + source, shape)
                join(
                    p + "input.weight",
                    tuple(p + part + "_proj.weight" for part in ("q", "k", "v")),
                )
                for part in ("q", "k"):
                    tensor(
                        p + part + "_norm.weight",
                        gg + "attn_" + part + "_norm.weight",
                        (c.head_dim,),
                        auxiliary=True,
                        offset_norm=True,
                    )
            p = hf + "mlp."
            if moe:
                tensor(
                    p + "gate.weight", gg + "ffn_gate_inp.weight", (c.experts, c.width)
                )
                tensor(
                    p + "shared_expert_gate.weight",
                    gg + "ffn_gate_inp_shexp.weight",
                    (1, c.width),
                    auxiliary=False,
                )
                for part, shape in (
                    ("gate", (c.experts, c.expert_width, c.width)),
                    ("up", (c.experts, c.expert_width, c.width)),
                    ("down", (c.experts, c.width, c.expert_width)),
                ):
                    tensor(
                        p + "experts." + part + "_proj.weight",
                        gg + "ffn_" + part + "_exps.weight",
                        shape,
                    )
                for part, shape in (
                    ("gate", (c.expert_width, c.width)),
                    ("up", (c.expert_width, c.width)),
                    ("down", (c.width, c.expert_width)),
                ):
                    tensor(
                        p + "shared_expert." + part + "_proj.weight",
                        gg + "ffn_" + part + "_shexp.weight",
                        shape,
                    )
                join(
                    p + "shared_expert.input.weight",
                    tuple(
                        p + "shared_expert." + part + "_proj.weight"
                        for part in ("gate", "up")
                    ),
                )
            else:
                for part, shape in (
                    ("gate", (c.intermediate, c.width)),
                    ("up", (c.intermediate, c.width)),
                    ("down", (c.width, c.intermediate)),
                ):
                    tensor(
                        p + part + "_proj.weight", gg + "ffn_" + part + ".weight", shape
                    )
                join(
                    p + "input.weight",
                    tuple(p + part + "_proj.weight" for part in ("gate", "up")),
                )
        unexpected = set(self.gguf.tensors) - self.used
        # MTP blocks are optional in GGUF and do not belong to text composition.
        extra = self.gguf.metadata.get(
            self.gguf.metadata["general.architecture"] + ".nextn_predict_layers", 0
        )
        if extra and not moe:
            gg = f"blk.{len(c.layers)}."
            hf = "mtp.layers.0."
            if gg + "nextn.eh_proj.weight" in self.gguf.tensors:
                self.mtp_used.add("token_embd.weight")
                self.mtp_groups[embedding] = self.groups[embedding]
                if self.head_name != embedding:
                    self.mtp_used.add("output.weight")
                    self.mtp_groups[self.head_name] = self.groups[self.head_name]
                for target, source in (
                    ("mtp.norm.weight", gg + "nextn.shared_head_norm.weight"),
                    ("mtp.pre_fc_norm_embedding.weight", gg + "nextn.enorm.weight"),
                    ("mtp.pre_fc_norm_hidden.weight", gg + "nextn.hnorm.weight"),
                    (hf + "input_layernorm.weight", gg + "attn_norm.weight"),
                    (
                        hf + "post_attention_layernorm.weight",
                        gg + "post_attention_norm.weight",
                    ),
                ):
                    tensor(
                        target,
                        source,
                        (c.width,),
                        auxiliary=True,
                        offset_norm=True,
                        mtp=True,
                    )
                tensor(
                    "mtp.fc.weight",
                    gg + "nextn.eh_proj.weight",
                    (c.width, 2 * c.width),
                    mtp=True,
                )
                p = hf + "self_attn."
                for target, source, shape in (
                    ("q_proj.weight", "attn_q.weight", (2 * c.heads * c.head_dim, c.width)),
                    ("k_proj.weight", "attn_k.weight", (c.kv_heads * c.head_dim, c.width)),
                    ("v_proj.weight", "attn_v.weight", (c.kv_heads * c.head_dim, c.width)),
                    ("o_proj.weight", "attn_output.weight", (c.width, c.heads * c.head_dim)),
                ):
                    tensor(p + target, gg + source, shape, mtp=True)
                join(
                    p + "input.weight",
                    tuple(p + part + "_proj.weight" for part in ("q", "k", "v")),
                    mtp=True,
                )
                for part in ("q", "k"):
                    tensor(
                        p + part + "_norm.weight",
                        gg + "attn_" + part + "_norm.weight",
                        (c.head_dim,),
                        auxiliary=True,
                        offset_norm=True,
                        mtp=True,
                    )
                p = hf + "mlp."
                for part, shape in (
                    ("gate", (c.intermediate, c.width)),
                    ("up", (c.intermediate, c.width)),
                    ("down", (c.width, c.intermediate)),
                ):
                    tensor(
                        p + part + "_proj.weight",
                        gg + "ffn_" + part + ".weight",
                        shape,
                        mtp=True,
                    )
                join(
                    p + "input.weight",
                    tuple(p + part + "_proj.weight" for part in ("gate", "up")),
                    mtp=True,
                )
        allowed = {
            name
            for name in unexpected
            if extra
            and name.startswith(
                tuple(f"blk.{i}." for i in range(len(c.layers), len(c.layers) + extra))
            )
        }
        if unexpected - allowed:
            raise GGUFError(
                f"unexpected Qwen GGUF tensors: {sorted(unexpected - allowed)[:4]}"
            )

    def graph_weights(self, graph, *, provider, branch="text"):
        if branch not in ("text", "mtp"):
            raise GGUFError("Qwen GGUF branch must be text or mtp")
        if branch == "mtp" and not self.mtp_groups:
            raise GGUFError("Qwen GGUF MTP is not implemented")
        from akas.quantization import get
        from akas.quantization.weights import Weight

        used = self.mtp_used if branch == "mtp" else self.used
        groups = self.mtp_groups if branch == "mtp" else self.groups
        buffers = {}
        for name in sorted(used):
            info = self.gguf.tensors[name]
            if name in self.auxiliary:
                shape, dtype, *_ = self.auxiliary[name]
                if provider == "webgpu" and dtype == "bfloat16":
                    dtype = "float16"
            else:
                storage = get("weights", "ggml").storage(
                    self.physical_shapes[name], provider=provider, encoding=info.type
                )
                shape, dtype = storage.values.shape, storage.values.dtype
            buffers[name] = graph.buffer(name, shape, dtype, role="weight")
        result = {}
        for name, sources in groups.items():
            if len(sources) == 1 and sources[0] in self.auxiliary:
                result[name] = buffers[sources[0]]
            else:
                parts = tuple(
                    dict(
                        rows=prod(self.physical_shapes[source][:-1]),
                        format="ggml",
                        options={"encoding": self.gguf.tensors[source].type},
                        row_order=self.orders[source][0],
                        column_order=self.orders[source][1],
                    )
                    for source in sources
                )
                result[name] = Weight(
                    self.group_shapes[name],
                    tuple(buffers[source] for source in sources),
                    parts,
                )
        return result

    def read(self, name):
        """Decode a mapped tensor for inspection, restoring native Qwen ordering."""
        import numpy as np
        from akas.quantization._shared.access import restore_order

        sources = self.groups.get(name, (name,))
        arrays = []
        for source in sources:
            values = self.gguf.array(source)
            row_order, column_order = self.orders[source]
            values = restore_order(values, 0, row_order)
            if column_order is not None:
                values = restore_order(values, -1, column_order)
            if source in self.auxiliary:
                shape, _, offset_norm, logarithm = self.auxiliary[source]
                if offset_norm:
                    values = values - np.float32(1)
                if logarithm:
                    if not np.isfinite(values).all() or np.any(values >= 0):
                        raise GGUFError(
                            f"{source}: expected finite negative SSM A values"
                        )
                    values = np.log(-values)
                values = values.reshape(shape)
            arrays.append(values)
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)

    def program_upload_options(self, weights, *, model_options):
        if set(weights) == self.used:
            return {}
        if self.mtp_used and set(weights) == (self.used | self.mtp_used):
            return {"branch": "all"}
        raise GGUFError("compiled Qwen GGUF weights differ from the text inventory")

    def upload(self, device, *, branch="text"):
        import numpy as np
        from akas.quantization._shared.packing import pack_words

        if branch not in ("text", "mtp", "all"):
            raise GGUFError("invalid Qwen GGUF upload branch")
        wanted = (
            self.used
            if branch == "text"
            else (self.used | self.mtp_used)
            if branch == "all"
            else self.mtp_used
        )
        result = {}
        try:
            for name in sorted(wanted):
                info = self.gguf.tensors[name]
                if name in self.auxiliary:
                    values = self.read(name)
                    dtype = self.auxiliary[name][1]
                    if device.info["provider"] == "webgpu" and dtype == "bfloat16":
                        result[name] = device.from_numpy(
                            np.ascontiguousarray(values.astype("float16"))
                        )
                    else:
                        result[name] = device.from_numpy(
                            values, **({"dtype": dtype} if dtype == "bfloat16" else {})
                        )
                else:
                    raw = self.gguf.packed(name)
                    if info.type in (0, 1):
                        raw = raw.view("float32" if info.type == 0 else "float16")
                    elif device.info["provider"] == "webgpu":
                        raw = pack_words(raw)
                    result[name] = device.from_numpy(np.ascontiguousarray(raw))
            return result
        except BaseException:
            for buffer in result.values():
                buffer.release()
            raise

    def inventory(self):
        from collections import Counter

        return dict(
            format="gguf",
            tensors=len(self.tensors),
            encodings=dict(Counter(info.encoding for info in self.tensors.values())),
            weight_bytes=sum(info.nbytes for info in self.tensors.values()),
        )
