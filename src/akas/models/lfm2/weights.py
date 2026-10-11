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

    def upload(self, device):
        return upload_packed(device, self.gguf)


def validate_tensors(config, gguf):
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
    if "output.weight" in gguf.tensors:
        expected["output.weight"] = (config.vocab, config.width)
    if set(expected) != set(gguf.tensors):
        raise GGUFError("unsupported/missing LFM2 tensor inventory")
    for name, shape in expected.items():
        if gguf.tensors[name].shape != shape:
            raise GGUFError(f"{name}: incompatible LFM2 tensor shape")
        if len(shape) == 1 or name.endswith("conv.weight"):
            if gguf.tensors[name].type != 0:
                raise GGUFError(f"{name}: requires float32 auxiliary weights")
