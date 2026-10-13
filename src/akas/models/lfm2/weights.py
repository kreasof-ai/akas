"""Validate an LFM2 GGUF checkpoint and upload its packed weights."""

from akas.checkpoints.upload import upload_packed
from akas.checkpoints.gguf import GGUF
from akas.checkpoints.gguf import GGUFError
from akas.models.lfm2.config import Config
from akas.text.tokenizer import Tokenizer


class LFM2Checkpoint:
    def __init__(self, path):
        self.gguf = GGUF(path)
        self.config = Config.from_gguf(self.gguf)
        self.tokenizer = Tokenizer.from_gguf(self.gguf.metadata)

    def inventory(self):
        from collections import Counter

        return dict(
            tensors=len(self.gguf.tensors),
            encodings=dict(Counter(t.type for t in self.gguf.tensors.values())),
        )

    def upload(self, device):
        return upload_packed(device, self.gguf)


def tensor_shapes(config, *, output=False):
    expected = {
        "token_embd.weight": (config.vocab, config.width),
        "token_embd_norm.weight": (config.width,),
    }
    for layer, kind in enumerate(config.layers):
        p = f"blk.{layer}."
        c = config.width
        f = config.ff
        d = config.head_dim
        kh = config.kv_heads
        expected.update(
            {
                p + "attn_norm.weight": (c,),
                p + "ffn_norm.weight": (c,),
                p + "ffn_gate.weight": (f, c),
                p + "ffn_up.weight": (f, c),
                p + "ffn_down.weight": (c, f),
            }
        )
        if kind == "conv":
            expected.update(
                {
                    p + "shortconv.in_proj.weight": (3 * c, c),
                    p + "shortconv.out_proj.weight": (c, c),
                    p + "shortconv.conv.weight": (c, 3),
                }
            )
        else:
            expected.update(
                {
                    p + "attn_q.weight": (c, c),
                    p + "attn_k.weight": (kh * d, c),
                    p + "attn_v.weight": (kh * d, c),
                    p + "attn_output.weight": (c, c),
                    p + "attn_q_norm.weight": (d,),
                    p + "attn_k_norm.weight": (d,),
                }
            )
    if output:
        expected["output.weight"] = (config.vocab, config.width)
    return expected


def validate_tensors(config, gguf):
    expected = tensor_shapes(config, output="output.weight" in gguf.tensors)
    if set(expected) != set(gguf.tensors):
        raise GGUFError("unsupported/missing LFM2 tensor inventory")
    for name, shape in expected.items():
        if gguf.tensors[name].shape != shape:
            raise GGUFError(f"{name}: incompatible LFM2 tensor shape")
        if len(shape) == 1 or name.endswith("conv.weight"):
            if gguf.tensors[name].type != 0:
                raise GGUFError(f"{name}: requires float32 auxiliary weights")


class ReferenceCheckpoint:
    """Canonical floating weights for the readable batched reference path.

    GGUF quantized matrices are dequantized once at load time. The existing
    packed path continues to use their original compressed storage.
    """

    @property
    def definition_options(self):
        return {"output": "output.weight" in self.names}

    def program_upload_options(self, weights, *, model_options):
        matrices = tensor_shapes(self.config, output="output.weight" in self.names)
        dtypes = {spec.dtype for name, spec in weights.items()
                  if len(matrices[name.split(".part")[0]]) == 2
                  and not name.endswith("conv.weight")}
        if len(dtypes) != 1:
            raise ValueError("floating checkpoint requires one matrix storage precision")
        return dict(matrix_dtype=dtypes.pop(),
                    max_weight_bytes=model_options.get("max_weight_bytes", 64 * 1024 * 1024))

    def __init__(self, path, *, tokenizer=True):
        from pathlib import Path
        import json
        from akas.checkpoints.safetensors import tensor_index, inspect_float_tensors

        self.path = Path(path)
        self.gguf = None
        if self.path.is_file():
            self.gguf = GGUF(self.path)
            self.config = Config.from_gguf(self.gguf)
            self.names = {name: name for name in self.gguf.tensors}
            self.tokenizer = (
                Tokenizer.from_gguf(self.gguf.metadata) if tokenizer else None
            )
        else:
            self.config = Config.from_huggingface(
                json.loads((self.path / "config.json").read_text())
            )
            index = tensor_index(self.path)
            self.names = huggingface_names(
                self.config, output="lm_head.weight" in index
            )
            shapes = tensor_shapes(self.config, output="output.weight" in self.names)
            hf_shapes = {self.names[name]: shape for name, shape in shapes.items()}
            for name in list(hf_shapes):
                if name.endswith("conv.weight"):
                    hf_shapes[name] = (self.config.width, 1, self.config.conv)
            self.tensors = inspect_float_tensors(self.path, index, hf_shapes)
            self.tokenizer = Tokenizer.from_pretrained(self.path) if tokenizer else None

    def read(self, name):
        from akas.checkpoints.safetensors import read_float_tensor

        if self.gguf is not None:
            return self.gguf.array(name).astype("float32")
        return read_float_tensor(self.tensors[self.names[name]]).reshape(
            tensor_shapes(self.config, output="output.weight" in self.names)[name]
        )

    def upload(
        self, device, *, max_weight_bytes=64 * 1024 * 1024, matrix_dtype="float32"
    ):
        from akas.checkpoints.partition import partitions

        result = {}
        try:
            for name in self.names:
                values = self.read(name)
                if values.ndim == 2 and not name.endswith("conv.weight"):
                    values = values.astype(matrix_dtype)
                for part in partitions(name, values.shape, max_bytes=max_weight_bytes):
                    selected = (
                        values
                        if part.name == name
                        else values[part.start : part.start + part.rows]
                    )
                    result[part.name] = device.from_numpy(selected.reshape(-1))
            return result
        except BaseException:
            for buffer in result.values():
                buffer.release()
            raise


def huggingface_names(config, *, output=False):
    names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "token_embd_norm.weight": "model.embedding_norm.weight",
    }
    if output:
        names["output.weight"] = "lm_head.weight"
    for i, kind in enumerate(config.layers):
        prefix, hf = f"blk.{i}.", f"model.layers.{i}."
        mapped = {
            "attn_norm": "operator_norm",
            "ffn_norm": "ffn_norm",
            "ffn_gate": "feed_forward.w1",
            "ffn_up": "feed_forward.w3",
            "ffn_down": "feed_forward.w2",
        }
        if kind == "conv":
            mapped.update(
                {
                    f"shortconv.{part}": f"conv.{part}"
                    for part in ("in_proj", "out_proj", "conv")
                }
            )
        else:
            mapped.update(
                {
                    "attn_q": "self_attn.q_proj",
                    "attn_k": "self_attn.k_proj",
                    "attn_v": "self_attn.v_proj",
                    "attn_output": "self_attn.out_proj",
                    "attn_q_norm": "self_attn.q_layernorm",
                    "attn_k_norm": "self_attn.k_layernorm",
                }
            )
        names.update(
            {
                prefix + key + ".weight": hf + value + ".weight"
                for key, value in mapped.items()
            }
        )
    return names
