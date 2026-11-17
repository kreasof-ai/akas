"""Read the packed ZONOS3-Mini backbone and F32 projector."""

from collections import Counter
from contextlib import ExitStack
from functools import cached_property
from math import prod
from types import SimpleNamespace

from akas.checkpoints.gguf import GGUF
from akas.models.zonos3.config import Zonos3MiniConfig


CODEBOOK_HEAD = "a.gen.code.head.weight"
CODEBOOK_EMBEDDING = "a.gen.code.out_embd.weight"
TOKEN_EMBEDDING = "token_embd.weight"


def _backbone_tensors(c):
    """Expected logical shapes and GGML types for the supported checkpoint."""
    width = c.width
    shapes = {
        TOKEN_EMBEDDING: ((c.vocab, width), 8),
        "output_norm.weight": ((width,), 0),
        "output_res_score.weight": ((width,), 0),
    }
    for index, recurrent in enumerate(c.recurrent):
        prefix = f"blk.{index}."
        layer = {
            name + ".weight": ((width,), 0)
            for name in ("attn_norm", "attn_res_score", "ffn_norm", "ffn_res_score")
        }
        if recurrent:
            delta = c.delta_width
            layer.update(
                {
                    "attn_qkv.weight": ((3 * delta, width), 8),
                    "attn_gate.weight": ((delta, width), 8),
                    "ssm_alpha.weight": ((c.delta_heads, width), 8),
                    "ssm_beta.weight": ((c.delta_heads, width), 8),
                    "ssm_conv1d.weight": ((3 * delta, c.conv_width), 0),
                    "ssm_dt.bias": ((c.delta_heads,), 0),
                    "ssm_a": ((c.delta_heads,), 0),
                    "ssm_norm.weight": ((c.delta_dim,), 0),
                    "ssm_out.weight": ((width, delta), 8),
                }
            )
        else:
            query, key = c.heads * c.head_dim, c.kv_heads * c.head_dim
            layer.update(
                {
                    "attn_q.weight": ((query, width), 8),
                    "attn_k.weight": ((key, width), 8),
                    "attn_v.weight": ((key, width), 8),
                    "attn_q_norm.weight": ((c.heads, c.head_dim), 0),
                    "attn_gate.weight": ((c.heads, width), 8),
                    "attn_output.weight": ((width, query), 8),
                }
            )
        if index < c.dense_layers:
            layer.update(
                {
                    "ffn_gate.weight": ((c.dense_width, width), 8),
                    "ffn_up.weight": ((c.dense_width, width), 8),
                    "ffn_down.weight": ((width, c.dense_width), 8),
                }
            )
        else:
            expert_in = (c.experts, c.expert_width, c.latent_width)
            layer.update(
                {
                    "ffn_gate_inp.weight": ((c.experts, width), 0),
                    "exp_probs_b.bias": ((c.experts,), 0),
                    "ffn_latent_down.weight": ((c.latent_width, width), 8),
                    "ffn_latent_up.weight": ((width, c.latent_width), 8),
                    "ffn_gate_exps.weight": (expert_in, 8),
                    "ffn_up_exps.weight": (expert_in, 8),
                    "ffn_down_exps.weight": (
                        (c.experts, c.latent_width, c.expert_width),
                        8,
                    ),
                }
            )
        shapes.update((prefix + name, spec) for name, spec in layer.items())
    return shapes


def _validate(gguf, expected):
    for name, (shape, encoding) in expected.items():
        info = gguf.tensors.get(name)
        if info is None or (info.shape, info.type) != (shape, encoding):
            raise ValueError(f"incompatible ZONOS3 tensor: {name}")


class _NativeTensorInfo:
    __slots__ = ("shape", "type", "encoding", "nbytes")

    def __init__(self, shape, kind, encoding, nbytes):
        self.shape = tuple(shape)
        self.type = kind
        self.encoding = encoding
        self.nbytes = int(nbytes)


class _NativeTensorStore:
    """Expose native PyTorch BF16/F32 tensors through the GGUF source contract."""

    def __init__(self, arrays, metadata=None):
        self._arrays = arrays
        self.metadata = metadata or {}
        self.tensors = {}
        for name, (raw, shape, kind) in arrays.items():
            encoding = "BF16" if kind == 30 else "F32"
            self.tensors[name] = _NativeTensorInfo(shape, kind, encoding, raw.nbytes)

    def packed(self, name):
        return self._arrays[name][0]

    def discard(self, name):
        return None

    def array(self, name, dtype=None):
        import numpy as np
        from tensor.runtime.dtypes import decode_bfloat16

        raw, shape, kind = self._arrays[name]
        if kind == 30:
            values = decode_bfloat16(raw.view(np.uint16)).reshape(shape)
        else:
            values = raw.view(np.float32).reshape(shape)
        return values.astype(dtype, copy=False) if dtype is not None else values

    def row(self, name, index):
        import numpy as np
        from tensor.runtime.dtypes import decode_bfloat16

        raw, shape, kind = self._arrays[name]
        cols = shape[-1]
        if kind == 30:
            words = raw.view(np.uint16).reshape(-1, cols)[index]
            return decode_bfloat16(words)
        return raw.view(np.float32).reshape(-1, cols)[index].copy()


def _wn_matrix(state, prefix):
    import torch

    g = state[prefix + ".weight_g"].float()
    v = state[prefix + ".weight_v"].float()
    norm = torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return (g * (v / norm)).squeeze(-1)


class NativeZonos3MiniCheckpoint:
    """Native BF16 ZONOS3-Mini LM directory paired with the FP32 keep2_plain codec."""

    storage_format = "bf16"

    def __init__(self, directory, codec=None):
        import hashlib
        import json
        import os
        from pathlib import Path

        import numpy as np
        import torch

        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise ValueError(
                f"native ZONOS3 checkpoint directory not found: {directory}"
            )
        params_raw = json.loads((self.directory / "params.json").read_text())
        self.params = params = params_raw["model"]
        self.config = c = Zonos3MiniConfig.from_zonos3(params_raw)

        spec = dict(params["text_tokenizer_spec"])
        tok_path = self.directory / spec["path"]
        tok_bytes = tok_path.read_bytes()
        if hashlib.sha256(tok_bytes).hexdigest() != spec["sha256"]:
            raise ValueError("ZONOS3 SuperBPE tokenizer SHA-256 mismatch")
        self._tokenizer_path = tok_path
        self._tokenizer_offset = int(spec["offset"])
        self._bos = int(spec["bos"])
        self._eos = int(spec["eos"])
        self.audio_pad_id = int(params["audio_pad_id"])
        self.text_vocab = int(params["text_vocab"])

        codec_path = Path(
            codec or os.environ.get("ZONOS3_COMPACT_CODEC", "")
        ).expanduser()
        if not codec_path.is_file():
            raise ValueError(
                "native ZONOS3 requires the keep2_plain compact codec checkpoint path"
            )
        self.codec_path = codec_path.resolve()

        weight_file = self.directory / "model.pth"
        if not weight_file.is_file():
            raise ValueError(f"missing ZONOS3 consolidated weights: {weight_file}")
        lm_raw = torch.load(
            weight_file, map_location="cpu", weights_only=False, mmap=True
        )
        state = (
            lm_raw["model"]
            if isinstance(lm_raw, dict) and "model" in lm_raw
            else lm_raw
        )

        def bf16_entry(tensor, expected_shape=None):
            t = tensor.detach().to(torch.bfloat16).contiguous()
            if expected_shape is not None and tuple(t.shape) != tuple(expected_shape):
                raise ValueError(
                    f"unexpected BF16 tensor shape {tuple(t.shape)} != {expected_shape}"
                )
            raw = np.frombuffer(t.view(torch.uint8).numpy().tobytes(), dtype=np.uint8)
            return (raw, tuple(t.shape), 30)

        def f32_entry(tensor, expected_shape=None):
            t = tensor.detach().to(torch.float32).contiguous()
            if expected_shape is not None and tuple(t.shape) != tuple(expected_shape):
                raise ValueError(
                    f"unexpected F32 tensor shape {tuple(t.shape)} != {expected_shape}"
                )
            raw = np.frombuffer(t.numpy().tobytes(), dtype=np.uint8)
            return (raw, tuple(t.shape), 0)

        queries = state["block_attnres_queries"]
        if tuple(queries.shape) != (2 * len(c.recurrent) + 1, c.width):
            raise ValueError("incompatible ZONOS3 block_attnres_queries shape")

        backbone = {
            TOKEN_EMBEDDING: bf16_entry(
                state[f"multi_embedder.embedders.{c.codebooks}.weight"],
                (c.vocab, c.width),
            ),
            "output_norm.weight": f32_entry(state["out_norm.weight"], (c.width,)),
            "output_res_score.weight": f32_entry(queries[-1], (c.width,)),
        }
        self.condition_tables = tuple(
            state[f"condition_embeddings.{i}.weight"].detach().float().numpy().copy()
            for i in range(len(c.condition_buckets))
        )
        cond_cat = torch.cat(
            [
                state[f"condition_embeddings.{i}.weight"].float()
                for i in range(len(c.condition_buckets))
            ],
            dim=0,
        )
        backbone["cond_embd.weight"] = f32_entry(
            cond_cat, (sum(b + 1 for b in c.condition_buckets), c.width)
        )
        self.audio_embedders = np.concatenate(
            [
                state[f"multi_embedder.embedders.{i}.weight"].detach().float().numpy()
                for i in range(c.codebooks)
            ],
            axis=0,
        ).copy()
        self.text_embedder = (
            state[f"multi_embedder.embedders.{c.codebooks}.weight"]
            .detach()
            .float()
            .numpy()
            .copy()
        )
        if "speaker_projection.weight" in state:
            self.speaker_weight = (
                state["speaker_projection.weight"].detach().float().numpy().copy()
            )
            self.speaker_bias = (
                state["speaker_projection.bias"].detach().float().numpy().copy()
            )
        else:
            self.speaker_weight = self.speaker_bias = None

        for idx, recurrent in enumerate(c.recurrent):
            p = f"blk.{idx}."
            src = f"layers.{idx}."
            backbone[p + "attn_norm.weight"] = f32_entry(
                state[src + "attention_norm.weight"], (c.width,)
            )
            backbone[p + "ffn_norm.weight"] = f32_entry(
                state[src + "ffn_norm.weight"], (c.width,)
            )
            backbone[p + "attn_res_score.weight"] = f32_entry(
                queries[2 * idx], (c.width,)
            )
            backbone[p + "ffn_res_score.weight"] = f32_entry(
                queries[2 * idx + 1], (c.width,)
            )
            if recurrent:
                delta = c.delta_width
                in_proj = state[src + "attention.in_proj.weight"]
                if tuple(in_proj.shape) != (
                    3 * delta + delta + 2 * c.delta_heads,
                    c.width,
                ):
                    raise ValueError(f"unexpected GDN in_proj shape at layer {idx}")
                qkv_w, gate_w, alpha_w, beta_w = in_proj.split(
                    [3 * delta, delta, c.delta_heads, c.delta_heads], dim=0
                )
                backbone[p + "input.weight"] = bf16_entry(in_proj)
                backbone[p + "attn_qkv.weight"] = bf16_entry(
                    qkv_w, (3 * delta, c.width)
                )
                backbone[p + "attn_gate.weight"] = bf16_entry(gate_w, (delta, c.width))
                backbone[p + "ssm_alpha.weight"] = bf16_entry(
                    alpha_w, (c.delta_heads, c.width)
                )
                backbone[p + "ssm_beta.weight"] = bf16_entry(
                    beta_w, (c.delta_heads, c.width)
                )
                backbone[p + "ssm_conv1d.weight"] = f32_entry(
                    state[src + "attention.conv.weight"].squeeze(1),
                    (3 * delta, c.conv_width),
                )
                backbone[p + "ssm_dt.bias"] = f32_entry(
                    state[src + "attention.dt_bias"], (c.delta_heads,)
                )
                backbone[p + "ssm_a_log"] = f32_entry(
                    state[src + "attention.A_log"], (c.delta_heads,)
                )
                backbone[p + "ssm_norm.weight"] = f32_entry(
                    state[src + "attention.out_weight"], (c.delta_dim,)
                )
                backbone[p + "ssm_out.weight"] = bf16_entry(
                    state[src + "attention.wo.weight"], (c.width, delta)
                )
            else:
                q_dim, kv_dim = c.heads * c.head_dim, c.kv_heads * c.head_dim
                wkv = state[src + "attention.wkv.weight"]
                if tuple(wkv.shape) != (2, kv_dim, c.width):
                    raise ValueError(f"unexpected attention wkv shape at layer {idx}")
                temp = (
                    state[src + "attention.temp"]
                    .detach()
                    .float()
                    .abs()
                    .reshape(c.heads, 1)
                    .expand(c.heads, c.head_dim)
                )
                backbone[p + "attn_q.weight"] = bf16_entry(
                    state[src + "attention.wq.weight"], (q_dim, c.width)
                )
                backbone[p + "attn_k.weight"] = bf16_entry(wkv[0], (kv_dim, c.width))
                backbone[p + "attn_v.weight"] = bf16_entry(wkv[1], (kv_dim, c.width))
                backbone[p + "attn_q_norm.weight"] = f32_entry(
                    temp, (c.heads, c.head_dim)
                )
                backbone[p + "attn_gate.weight"] = bf16_entry(
                    state[src + "attention.gater.weight"], (c.heads, c.width)
                )
                backbone[p + "attn_output.weight"] = bf16_entry(
                    state[src + "attention.wo.weight"], (c.width, q_dim)
                )
            if idx < c.dense_layers:
                w_in = state[src + "feed_forward.w_in.weight"]
                if tuple(w_in.shape) != (2, c.dense_width, c.width):
                    raise ValueError(f"unexpected dense w_in shape at layer {idx}")
                backbone[p + "ffn_up.weight"] = bf16_entry(
                    w_in[0], (c.dense_width, c.width)
                )
                backbone[p + "ffn_gate.weight"] = bf16_entry(
                    w_in[1], (c.dense_width, c.width)
                )
                backbone[p + "ffn_down.weight"] = bf16_entry(
                    state[src + "feed_forward.w_out.weight"], (c.width, c.dense_width)
                )
            else:
                gate_up = state[src + "feed_forward.experts.gate_up_proj"]
                if tuple(gate_up.shape) != (
                    c.experts,
                    2 * c.expert_width,
                    c.latent_width,
                ):
                    raise ValueError(
                        f"unexpected expert gate_up_proj shape at layer {idx}"
                    )
                backbone[p + "ffn_route_norm.weight"] = f32_entry(
                    state[src + "feed_forward.router.rmsnorm_eda.weight"], (c.width,)
                )
                backbone[p + "ffn_gate_inp.weight"] = bf16_entry(
                    state[src + "feed_forward.router.router_linear.weight"],
                    (c.experts, c.width),
                )
                backbone[p + "exp_probs_b.bias"] = f32_entry(
                    -state[src + "feed_forward.router.balancing_biases"].float(),
                    (c.experts,),
                )
                backbone[p + "ffn_latent_down.weight"] = bf16_entry(
                    state[src + "feed_forward.latent_down.weight"],
                    (c.latent_width, c.width),
                )
                backbone[p + "ffn_latent_up.weight"] = bf16_entry(
                    state[src + "feed_forward.latent_up.weight"],
                    (c.width, c.latent_width),
                )
                backbone[p + "ffn_gate_exps.weight"] = bf16_entry(
                    gate_up[:, : c.expert_width, :],
                    (c.experts, c.expert_width, c.latent_width),
                )
                backbone[p + "ffn_up_exps.weight"] = bf16_entry(
                    gate_up[:, c.expert_width :, :],
                    (c.experts, c.expert_width, c.latent_width),
                )
                backbone[p + "ffn_down_exps.weight"] = bf16_entry(
                    state[src + "feed_forward.experts.down_proj"],
                    (c.experts, c.latent_width, c.expert_width),
                )

        codec_blob = torch.load(
            self.codec_path, map_location="cpu", weights_only=False, mmap=True
        )
        if (
            codec_blob.get("format") != "zonos3-istft-codec-v1"
            or codec_blob.get("decoder") != "keep2_plain"
        ):
            raise ValueError(
                "requires the keep2_plain zonos3-istft-codec-v1 checkpoint"
            )
        self.codec_config = codec_blob["config"]
        cs = codec_blob["state_dict"]

        code_shape = (c.codebooks * c.code_vocab, c.width)
        embed_cat = torch.cat(
            [state[f"multi_embedder.embedders.{i}.weight"] for i in range(c.codebooks)],
            dim=0,
        )
        projector = {
            CODEBOOK_HEAD: bf16_entry(state["multi_output.weight"], code_shape),
            CODEBOOK_EMBEDDING: f32_entry(embed_cat, code_shape),
            "a.gen.code.out_embd.bf16": bf16_entry(embed_cat, code_shape),
        }
        sem_cb = cs["quantizer.semantic_quantizer.quantizers.0.codebook.weight"].float()
        sem_proj = _wn_matrix(cs, "quantizer.semantic_quantizer.quantizers.0.out_proj")
        projector["a.gen.wav.sem.table"] = f32_entry(
            sem_cb @ sem_proj.T, (c.codes, 768)
        )
        projector["a.gen.wav.sem.bias"] = f32_entry(
            cs["quantizer.semantic_quantizer.quantizers.0.out_proj.bias"], (768,)
        )
        for b in range(4):
            src = f"semantic_head.{b}."
            dst = f"a.gen.wav.sem.{b}."
            projector[dst + "dw.weight"] = f32_entry(cs[src + "dwconv.weight"])
            projector[dst + "dw.bias"] = f32_entry(cs[src + "dwconv.bias"])
            projector[dst + "norm.weight"] = f32_entry(cs[src + "norm.weight"])
            projector[dst + "norm.bias"] = f32_entry(cs[src + "norm.bias"])
            projector[dst + "pw1.weight"] = f32_entry(cs[src + "pwconv1.weight"])
            projector[dst + "pw1.bias"] = f32_entry(cs[src + "pwconv1.bias"])
            projector[dst + "pw2.weight"] = f32_entry(cs[src + "pwconv2.weight"])
            projector[dst + "pw2.bias"] = f32_entry(cs[src + "pwconv2.bias"])
        projector["a.gen.wav.sem.out.weight"] = f32_entry(
            _wn_matrix(cs, "semantic_head.4"), (1024, 768)
        )
        projector["a.gen.wav.sem.out.bias"] = f32_entry(
            cs["semantic_head.4.bias"], (1024,)
        )
        res_tables, res_biases = [], []
        for q_idx in range(c.codebooks - 1):
            q_src = f"quantizer.quantizer.quantizers.{q_idx}"
            q_cb = cs[q_src + ".codebook.weight"].float()
            q_proj = _wn_matrix(cs, q_src + ".out_proj")
            res_tables.append(q_cb @ q_proj.T)
            res_biases.append(cs[q_src + ".out_proj.bias"].float())
        projector["a.gen.wav.res.table"] = f32_entry(
            torch.stack(res_tables, dim=0), (c.codebooks - 1, c.codes, 1024)
        )
        projector["a.gen.wav.res.bias"] = f32_entry(
            torch.stack(res_biases, dim=0), (c.codebooks - 1, 1024)
        )
        for stage in range(5):
            src_stage = f"stages.{2 * stage}."
            dst_stage = f"a.gen.wav.stage.{stage}."
            projector[dst_stage + "in.weight"] = f32_entry(
                cs[src_stage + "input_proj.weight"]
            )
            for blk in range(2):
                sb = f"{src_stage}layers.{blk}."
                db = f"{dst_stage}blk.{blk}."
                projector[db + "norm1.weight"] = f32_entry(cs[sb + "norm1.weight"])
                projector[db + "norm1.bias"] = f32_entry(cs[sb + "norm1.bias"])
                projector[db + "qkv.weight"] = f32_entry(cs[sb + "qkv.weight"])
                projector[db + "wo.weight"] = f32_entry(cs[sb + "attention_out.weight"])
                projector[db + "attn_scale"] = f32_entry(cs[sb + "attention_scale"])
                projector[db + "norm2.weight"] = f32_entry(cs[sb + "norm2.weight"])
                projector[db + "norm2.bias"] = f32_entry(cs[sb + "norm2.bias"])
                projector[db + "ffn1.weight"] = f32_entry(cs[sb + "ffn.0.weight"])
                projector[db + "ffn2.weight"] = f32_entry(cs[sb + "ffn.3.weight"])
                projector[db + "ffn_scale"] = f32_entry(cs[sb + "ffn_scale"])
            if stage < 4:
                projector[dst_stage + "out.weight"] = f32_entry(
                    cs[src_stage + "output_proj.weight"]
                )
        for b in range(8):
            src = f"head.blocks.{b}."
            dst = f"a.gen.wav.head.{b}."
            projector[dst + "dw.weight"] = f32_entry(cs[src + "dwconv.weight"])
            projector[dst + "dw.bias"] = f32_entry(cs[src + "dwconv.bias"])
            projector[dst + "norm.weight"] = f32_entry(cs[src + "norm.weight"])
            projector[dst + "norm.bias"] = f32_entry(cs[src + "norm.bias"])
            projector[dst + "pw1.weight"] = f32_entry(cs[src + "pwconv1.weight"])
            projector[dst + "pw1.bias"] = f32_entry(cs[src + "pwconv1.bias"])
            projector[dst + "pw2.weight"] = f32_entry(cs[src + "pwconv2.weight"])
            projector[dst + "pw2.bias"] = f32_entry(cs[src + "pwconv2.bias"])
        projector["a.gen.wav.head.norm.weight"] = f32_entry(cs["head.norm.weight"])
        projector["a.gen.wav.head.norm.bias"] = f32_entry(cs["head.norm.bias"])
        projector["a.gen.wav.head.out.weight"] = f32_entry(
            torch.cat([cs["head.outs.0.weight"], cs["head.outs.1.weight"]], dim=0)
        )
        projector["a.gen.wav.head.out.bias"] = f32_entry(
            torch.cat([cs["head.outs.0.bias"], cs["head.outs.1.bias"]], dim=0)
        )
        projector["a.gen.wav.mixer.weight"] = f32_entry(
            cs["head.mixer.weight"][0], (2, 127)
        )
        projector["a.gen.wav.resample.weight"] = f32_entry(
            cs["to_external.kernel"][:, 0, :], (735, 770)
        )
        for idx, (n_fft, env_div) in enumerate(((736, 1.5), (2208, 4.5))):
            bins = n_fft // 2 + 1
            win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)) / env_div
            t_grid = np.arange(n_fft, dtype=np.float64)[:, None]
            k_grid = np.arange(bins, dtype=np.float64)[None, :]
            scale = np.full((1, bins), 2.0 / n_fft, dtype=np.float64)
            scale[0, 0] = 1.0 / n_fft
            scale[0, -1] = 1.0 / n_fft
            re_b = win[:, None] * scale * np.cos(2.0 * np.pi * t_grid * k_grid / n_fft)
            im_b = -win[:, None] * scale * np.sin(2.0 * np.pi * t_grid * k_grid / n_fft)
            im_b[:, 0] = 0.0
            im_b[:, -1] = 0.0
            basis = torch.from_numpy(
                np.concatenate((re_b, im_b), axis=1).astype(np.float32)
            )
            projector[f"a.gen.wav.fourier{idx}.weight"] = f32_entry(
                basis, (n_fft, 2 * bins)
            )
        rope_pos = np.arange(8192, dtype=np.float64)[:, None]
        rope_ang = rope_pos * (
            10000.0 ** (-np.arange(0, 64, 2, dtype=np.float64) / 64.0)
        )
        codec_rope = torch.from_numpy(
            np.concatenate((np.cos(rope_ang), np.sin(rope_ang)), axis=1).astype(
                np.float32
            )
        )
        projector["codec_rope.weight"] = f32_entry(codec_rope, (8192, 64))

        self.backbone = _NativeTensorStore(backbone)
        self.projector = _NativeTensorStore(
            projector, metadata={"clip.gen.audio.model_variant": "keep2_plain"}
        )

    def source(self, name):
        return self.backbone if name in self.backbone.tensors else self.projector

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_file(self._tokenizer_path)

    def text_token_ids(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ZONOS3 speech input is empty")
        raw = self.tokenizer.encode(text, add_bos=False)
        return [self._bos, *(t + self._tokenizer_offset for t in raw), self._eos]

    def prompt_frames(self, text, *, silence_frames=3, condition_buckets=None):
        """Build the exact [condition_rows | text_rows | silence_rows] (N, 11) int32 prompt."""
        import numpy as np

        from akas.models.zonos3.model import SILENCE

        c = self.config
        cond_vals = (
            tuple(c.condition_buckets)
            if condition_buckets is None
            else tuple(int(v) for v in condition_buckets)
        )
        if len(cond_vals) != len(c.condition_buckets):
            raise ValueError("invalid ZONOS3 condition bucket count")
        for idx, val in enumerate(cond_vals):
            if not 0 <= val <= c.condition_buckets[idx]:
                raise ValueError(f"out-of-range condition bucket {val} at row {idx}")
        text_ids = self.text_token_ids(text)
        cond_rows = np.full(
            (len(cond_vals), c.codebooks + 1), self.audio_pad_id, dtype="int32"
        )
        cond_rows[:, c.codebooks] = self.text_vocab
        txt_rows = np.full(
            (len(text_ids), c.codebooks + 1), self.audio_pad_id, dtype="int32"
        )
        txt_rows[:, c.codebooks] = np.asarray(text_ids, dtype="int32")
        sil_rows = np.full(
            (silence_frames, c.codebooks + 1), self.audio_pad_id, dtype="int32"
        )
        sil_rows[:, c.codebooks] = self.text_vocab
        for r in range(silence_frames):
            limit = min(r + 1, c.codebooks)
            sil_rows[r, :limit] = SILENCE[:limit]
        return np.concatenate((cond_rows, txt_rows, sil_rows), axis=0), cond_vals

    def prompt_encoded_frames(self, text, *, silence_frames=3, condition_buckets=None):
        """Return (N, 11) int32 frames with negative condition-row markers for zonos3_frame_embed."""
        frames, cond_vals = self.prompt_frames(
            text, silence_frames=silence_frames, condition_buckets=condition_buckets
        )
        encoded = frames.copy()
        offset = 0
        for i, val in enumerate(cond_vals):
            encoded[i, self.config.codebooks] = -(1 + offset + int(val))
            offset += self.config.condition_buckets[i] + 1
        return encoded, cond_vals

    def embed_frames(self, frames, *, condition_values=None, condition_start=0):
        """Compute exact pre-emb_norm embeddings matching Zonos3ForCausalLM.embed."""
        import numpy as np

        c = self.config
        arr = np.asarray(frames, dtype="int32")
        if arr.ndim == 1:
            arr = arr[None, :]
        offsets = np.arange(c.codebooks, dtype="int32") * c.code_vocab
        audio_ids = arr[:, : c.codebooks] + offsets[None, :]
        audio_sum = self.audio_embedders[audio_ids].sum(axis=1, dtype="float32")
        if arr.shape[1] > c.codebooks:
            text_emb = self.text_embedder[arr[:, c.codebooks]]
            out = (audio_sum + text_emb).astype("float32")
        else:
            out = (audio_sum + self.text_embedder[self.text_vocab]).astype("float32")
        if condition_values is not None:
            for i, bucket in enumerate(condition_values):
                out[condition_start + i] = self.condition_tables[i][bucket]
        return out

    def graph_weight(self, graph, name, *, provider):
        from akas.quantization.weights.ggml.definition import weight_storage

        info = self.source(name).tensors[name]
        count, dtype = weight_storage(
            info.type, prod(info.shape), packed_words=provider == "webgpu"
        )
        return graph.buffer(name, (count,), dtype, role="weight")

    def upload(self, device, names, *, context=None):
        import numpy as np

        from akas.checkpoints.upload import upload_packed

        names = tuple(names)
        regular = [
            name
            for name in names
            if name
            not in ("rope", "unit.weight", "head_unit.weight", "codec_unit.weight")
        ]
        weights = {}
        with ExitStack() as cleanup:
            for source in (self.backbone, self.projector):
                tensors = {
                    name: source.tensors[name]
                    for name in regular
                    if self.source(name) is source
                }
                if tensors:
                    loaded = upload_packed(
                        device,
                        SimpleNamespace(
                            tensors=tensors,
                            packed=source.packed,
                            discard=source.discard,
                        ),
                    )
                    for value in loaded.values():
                        cleanup.callback(value.release)
                    weights.update(loaded)
            for name in names:
                if name in weights:
                    continue
                if name == "unit.weight":
                    value = np.ones(self.config.width, dtype="float32")
                elif name == "head_unit.weight":
                    value = np.ones(self.config.head_dim, dtype="float32")
                elif name == "codec_unit.weight":
                    value = np.ones(64, dtype="float32")
                elif name == "rope":
                    c = self.config
                    if type(context) is not int or not 1 <= context <= c.max_context:
                        raise ValueError(
                            "ZONOS3 rotary upload requires a valid context"
                        )
                    frequency = c.rope_theta ** (
                        -np.arange(0, c.head_dim, 2) / c.head_dim
                    )
                    angle = np.arange(context)[:, None] * frequency
                    value = np.concatenate(
                        (np.cos(angle), np.sin(angle)), axis=1
                    ).astype("float32")
                else:
                    raise ValueError(f"unknown ZONOS3 weight buffer: {name}")
                weights[name] = device.from_numpy(value)
                cleanup.callback(weights[name].release)
            cleanup.pop_all()
        return weights

    def inventory(self):
        return {
            label: dict(Counter(info.encoding for info in source.tensors.values()))
            for label, source in (
                ("backbone", self.backbone),
                ("projector", self.projector),
            )
        }


class Zonos3MiniCheckpoint:
    """The matched GGUF pair or native BF16 directory for ZONOS3-Mini."""

    def __new__(cls, backbone, projector=None):
        from pathlib import Path

        if Path(backbone).is_dir():
            return NativeZonos3MiniCheckpoint(backbone, projector)
        return super().__new__(cls)

    def __init__(self, backbone, projector):
        self.backbone = GGUF(backbone)
        self.projector = GGUF(projector)
        self.config = c = Zonos3MiniConfig.from_gguf(self.backbone.metadata)
        if self.backbone.metadata.get("general.file_type") != 7:
            raise ValueError("requires the ZONOS3-Mini Q8_0 backbone")
        _validate(self.backbone, _backbone_tensors(c))
        metadata = self.projector.metadata
        if (
            metadata.get("general.architecture") != "clip"
            or metadata.get("clip.gen.audio.projector_type") != "zonos3_gen"
            or metadata.get("clip.gen.audio.model_variant") != "keep2_plain"
        ):
            raise ValueError("requires the ZONOS3-Mini keep2_plain projector")
        shape = (c.codebooks * c.code_vocab, c.width)
        _validate(
            self.projector, {CODEBOOK_HEAD: (shape, 0), CODEBOOK_EMBEDDING: (shape, 0)}
        )

    def source(self, name):
        return self.backbone if name in self.backbone.tensors else self.projector

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_gguf(self.backbone.metadata)

    def prompt_ids(self, text):
        """Encode normalized text with the default conditioning and boundaries."""
        if not text.strip():
            raise ValueError("ZONOS3 speech input is empty")
        tokenizer = self.tokenizer
        conditions = ("loudness_unk", "fmax_unk", "speaking_rate_unk")
        return [
            *(tokenizer.ids[f"<|{name}|>"] for name in conditions),
            *tokenizer.encode(text, add_bos=True),
            tokenizer.eos,
        ]

    @cached_property
    def default_voices(self):
        """Select the first shipped preset for each language, preferring the GGUF default."""
        from akas.models.zonos3.config import LANGUAGES

        default = self.projector.metadata.get("clip.gen.audio.default_voice")
        voices = {}
        for name in self.projector.tensors:
            voice = name.removeprefix("voice.")
            if voice == name:
                continue
            tag = voice.split("_", 2)
            if len(tag) < 3:
                continue
            language = tag[1].partition("-")[0]
            if language in LANGUAGES and (voice == default or language not in voices):
                voices[language] = voice
        return voices

    def voice_rows(self, voice=None, *, lang="en"):
        if voice is None:
            voice = self.default_voices.get(
                lang, self.projector.metadata.get("clip.gen.audio.default_voice")
            )
        name = f"voice.{voice}"
        info = self.projector.tensors.get(name)
        if (
            info is None
            or info.type != 0
            or len(info.shape) != 2
            or info.shape[1] != self.config.width
        ):
            raise ValueError(f"invalid ZONOS3 predefined voice: {voice}")
        return self.projector.array(name)

    def graph_weight(self, graph, name, *, provider):
        from akas.quantization.weights.ggml.definition import weight_storage

        info = self.source(name).tensors[name]
        count, dtype = weight_storage(
            info.type, prod(info.shape), packed_words=provider == "webgpu"
        )
        return graph.buffer(name, (count,), dtype, role="weight")

    def upload(self, device, names, *, context=None):
        import numpy as np

        from akas.checkpoints.upload import upload_packed

        names = tuple(names)
        regular = [
            name
            for name in names
            if name not in ("rope", "unit.weight", "head_unit.weight")
            and not name.endswith(".ssm_a_log")
        ]
        weights = {}
        with ExitStack() as cleanup:
            for source in (self.backbone, self.projector):
                tensors = {
                    name: source.tensors[name]
                    for name in regular
                    if self.source(name) is source
                }
                if tensors:
                    loaded = upload_packed(
                        device,
                        SimpleNamespace(
                            tensors=tensors,
                            packed=source.packed,
                            discard=source.discard,
                        ),
                    )
                    for value in loaded.values():
                        cleanup.callback(value.release)
                    weights.update(loaded)
            for name in names:
                if name in weights:
                    continue
                if name == "unit.weight":
                    value = np.ones(self.config.width, dtype="float32")
                elif name == "head_unit.weight":
                    value = np.ones(self.config.head_dim, dtype="float32")
                elif name == "rope":
                    c = self.config
                    if type(context) is not int or not 1 <= context <= c.max_context:
                        raise ValueError(
                            "ZONOS3 rotary upload requires a valid context"
                        )
                    frequency = c.rope_theta ** (
                        -np.arange(0, c.head_dim, 2) / c.head_dim
                    )
                    angle = np.arange(context)[:, None] * frequency
                    value = np.concatenate(
                        (np.cos(angle), np.sin(angle)), axis=1
                    ).astype("float32")
                else:
                    decay = self.backbone.array(name.removesuffix("_log"))
                    if not np.all(np.isfinite(decay) & (decay < 0)):
                        raise ValueError(f"invalid ZONOS3 GDN decay: {name}")
                    value = np.log(-decay)
                weights[name] = device.from_numpy(value)
                cleanup.callback(weights[name].release)
            cleanup.pop_all()
        return weights

    def inventory(self):
        return {
            label: dict(Counter(info.encoding for info in source.tensors.values()))
            for label, source in (
                ("backbone", self.backbone),
                ("projector", self.projector),
            )
        }
