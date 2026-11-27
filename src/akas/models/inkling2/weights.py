"""Read the original Inkling 2 Mini text weights and packed expert metadata."""

from dataclasses import replace
from pathlib import Path

from akas.checkpoints.metadata import json_document
from akas.checkpoints.safetensors import inspect_tensors, tensor_index
from akas.models.inkling2.config import Inkling2MiniConfig
from akas.text.tokenizer import Tokenizer

_PREFIX = "model.llm."
_DTYPES = {"BF16": "bfloat16", "F32": "float32", "U8": "uint8", "F8_E4M3": "uint8"}


class _InklingTokenizer(Tokenizer):
    """Format a single user message with the checkpoint's default effort."""

    def chat(self, text):
        prefix = (
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|><|message_user|><|content_text|>"
        )
        return (
            self.encode(prefix, add_bos=False, parse_special=True)
            + self.encode(text, add_bos=False)
            + self.encode("<|end_message|>", add_bos=False, parse_special=True)
        )


def _expert_formats(directory):
    data = json_document((directory / "hf_quant_config.json").read_text())
    result = {}
    for group in data["model"]["parameter_groups"]:
        precision = group["compute_dtype"]
        if precision not in ("nvfp4", "mxfp8"):
            continue
        expected = (
            ("nvfp4_nearest", [1, 16])
            if precision == "nvfp4"
            else ("mxfp8_nearest", [32, 32])
        )
        if (group["weight_transform"], group["weight_block_shape"]) != expected:
            raise ValueError(f"unsupported Inkling 2 Mini {precision} encoding")
        for name in group["tensors"]:
            if name in result:
                raise ValueError(f"duplicate Inkling 2 Mini expert precision: {name}")
            result[name] = precision
    return result


def _tensors(config, formats):

    tensors = {}

    def add(name, shape, dtype="BF16"):
        tensors[_PREFIX + name] = (shape, dtype)

    width, experts, intermediate = config.width, config.experts, config.expert_width
    query, kv = config.heads * config.head_dim, config.kv_heads * config.head_dim
    add("embed.weight", (config.padded_vocab, width))
    add("unembed.weight", (config.padded_vocab, width))
    add("embed_norm.weight", (width,))
    add("norm.weight", (width,))
    add("attn_res.output_query", (width,), "F32")
    for name in (
        "middle_cycle_cross_pass_queries",
        "middle_cycle_cross_pass_kv_queries",
    ):
        add(name, (2 * len(config.loop), width), "F32")

    for layer in range(config.layers):
        prefix = f"layers.{layer}."
        local = layer in config.local_layers
        extent = config.window if local else config.relative_extent
        if layer:
            query_shape = (width,) if local else (3, width)
            add(f"attn_res.attn_queries.{layer - 1}", query_shape, "F32")
        add(f"attn_res.mlp_queries.{layer}", (width,), "F32")

        for name, shape in (
            ("attn.wq_du.weight", (query, width)),
            ("attn.wk_dv.weight", (kv, width)),
            ("attn.wv_dv.weight", (kv, width)),
            ("attn.wo_ud.weight", (width, query)),
            ("attn.gate_attn.weight", (query, width)),
            ("attn.attn_sink_proj.weight", (config.heads, width)),
            ("attn.wr_du.weight", (config.heads * config.relative_dim, width)),
            ("attn.rel_logits_proj.proj", (config.relative_dim, extent)),
            ("attn.q_l2_clip.scale", (config.heads, config.head_dim)),
            ("attn.rel_l2_clip.scale", (config.heads, config.relative_dim)),
            ("attn_sconv.weight", (width, 1, config.taps)),
            ("mlp_sconv.weight", (width, 1, config.taps)),
            ("mlp_norm.weight", (width,)),
            ("mlp.gate.weight", (experts, width)),
        ):
            add(prefix + name, shape)
        add(prefix + "mlp.gate.bias", (experts,), "F32")
        for stream in ("k", "v"):
            conv = prefix + f"attn.{stream}_sconv."
            if local:
                add(conv + "weight", (kv, 1, config.taps))
            else:
                add(conv + "static_w", (config.taps, kv))
                add(conv + "weight_proj", (config.kv_heads * config.taps, width))
        for name in (
            ("attn_norm",)
            if local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        ):
            add(prefix + name + ".weight", (width,))

        if layer in config.loop:
            rank = config.gate_rank
            for stream in ("attn", "k", "v", "mlp"):
                control = prefix + f"controllers.{stream}."
                add(control + "down_weight", (rank, 2 * width))
                add(control + "gain_weight", (width, rank))
                add(control + "forget_weight", (width, rank))

        bank = prefix + "mlp.experts."
        for part, columns, depth in (
            ("w13", 2 * intermediate, width),
            ("w2", width, intermediate),
        ):
            name = bank + part + "_weight"
            precision = formats.get(_PREFIX + name)
            if precision == "nvfp4" and depth % 16 == 0:
                add(name, (experts, columns, depth // 2), "U8")
                add(name + "_scale", (experts, columns, depth // 16), "F8_E4M3")
                add(name + "_amax", (experts,), "F32")
                add(name + "_outer_scale_denominator", (1,), "F32")
            elif precision == "mxfp8" and depth % 32 == 0 and columns % 32 == 0:
                add(name, (experts, columns, depth), "F8_E4M3")
                add(name + "_scale", (experts, columns, depth // 32), "U8")
            else:
                raise ValueError(f"unsupported or missing expert encoding: {name}")
    if formats.keys() - tensors.keys():
        raise ValueError("Inkling 2 Mini quantization metadata names unknown experts")
    return tensors


class Inkling2MiniCheckpoint:
    """Validate the text tensor contract and retain its native storage bytes."""

    weight_format = "nvfp4-mxfp8"

    def __init__(self, path, *, tokenizer=True):
        directory = Path(path)
        self.config = Inkling2MiniConfig.from_huggingface(
            json_document((directory / "config.json").read_text())
        )
        self._formats = _expert_formats(directory)
        contract = _tensors(self.config, self._formats)
        tensors = inspect_tensors(
            directory,
            tensor_index(directory),
            {name: item[0] for name, item in contract.items()},
            {name: item[1] for name, item in contract.items()},
        )
        # Output tensors use only the real vocabulary, excluding padded logits.
        head = _PREFIX + "unembed.weight"
        self._tensors = dict(tensors)
        self._tensors[head] = replace(
            tensors[head],
            shape=(self.config.vocab, self.config.width),
            nbytes=2 * self.config.vocab * self.config.width,
        )
        self.tokenizer = (
            _InklingTokenizer.from_pretrained(directory) if tokenizer else None
        )

    def inventory(self):
        from collections import Counter

        return dict(
            tensors=len(self._tensors),
            encodings=dict(Counter(item.dtype for item in self._tensors.values())),
        )

    def expert_format(self, name):
        return self._formats[_PREFIX + name]

    def graph_weight(self, graph, name):
        name = _PREFIX + name
        if name in graph.buffers:
            return graph.buffer_at(name)
        tensor = self._tensors[name]
        return graph.buffer(name, tensor.shape, _DTYPES[tensor.dtype], role="weight")

    def upload(self, device, *, weights=None):
        from akas.checkpoints.upload import upload_shards

        if device.info["provider"] != "cuda":
            raise ValueError("Inkling 2 Mini requires CUDA")
        wanted = tuple(self._tensors) if weights is None else tuple(weights)
        layouts = {name: (name, 0, self._tensors[name].shape) for name in wanted}
        return upload_shards(device, self._tensors, wanted, layouts)
