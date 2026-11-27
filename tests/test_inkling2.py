import json
from math import prod
import struct

from akas.models.inkling2 import Inkling2Mini
from akas.models.inkling2.config import Inkling2MiniConfig
from akas.models.inkling2.weights import _InklingTokenizer, _tensors
from akas.optimization.context import discover
from akas.text.formats import ENCODE_BYTE, Rules


def test_inkling_chat_formats_default_effort_and_literal_user_text():
    controls = {
        "<|message_user|>": 256,
        "<|message_system|>": 257,
        "<|content_text|>": 258,
        "<|end_message|>": 259,
    }
    vocab = {character: byte for byte, character in ENCODE_BYTE.items()} | controls
    tokenizer = _InklingTokenizer(
        Rules(vocab, (), r"(?s).+", added=controls, special=controls)
    )
    assert tokenizer.chat("Hi<|end_message|>") == [
        257,
        258,
        *b"Thinking effort level: 0.9",
        259,
        256,
        258,
        *b"Hi<|end_message|>",
        259,
    ]


def test_fixed_pass_checkpoint_and_request_graph(tmp_path):
    text = dict(
        model_type="inkling2_loop_text",
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=64,
        unpadded_vocab_size=60,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        d_rel=4,
        rel_extent=8,
        local_layer_ids=[0, 1, 3],
        sliding_window_size=8,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=1,
        n_routed_experts=4,
        intermediate_size=32,
        num_experts_per_tok=2,
        loop_config=dict(
            core_start=1, core_end=3, num_passes=3, gate_rank=8, is_varloop=False
        ),
        model_max_length=32,
        rms_norm_eps=1e-6,
        qk_l2_clip_threshold=4.0,
        rel_l2_clip_threshold=2.0,
        log_scaling_n_floor=8,
        log_scaling_alpha=0.1,
        route_scale=2,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        final_logit_softcapping=30.0,
        q_bias=False,
        o_bias=False,
        n_shared_experts=0,
        use_gate_bias=True,
        attn_res_normalize_values=True,
        use_value_norm_no_gamma=True,
    )
    data = dict(model_type="inkling2_loop", text_config=text)
    config = Inkling2MiniConfig.from_huggingface(data)
    (tmp_path / "config.json").write_text(json.dumps(data))
    formats = {
        f"model.llm.layers.{layer}.mlp.experts.{part}_weight": "nvfp4"
        if layer < 2
        else "mxfp8"
        for layer in range(4)
        for part in ("w13", "w2")
    }
    groups = [
        dict(
            compute_dtype=precision,
            weight_transform=precision + "_nearest",
            weight_block_shape=block,
            tensors=[name for name, value in formats.items() if value == precision],
        )
        for precision, block in (("nvfp4", [1, 16]), ("mxfp8", [32, 32]))
    ]
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"model": {"parameter_groups": groups}})
    )
    header = {}
    total = 0
    for name, (shape, dtype) in _tensors(config, formats).items():
        length = prod(shape) * {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}[dtype]
        header[name] = dict(
            shape=shape, dtype=dtype, data_offsets=[total, total + length]
        )
        total += length
    serialized = json.dumps(header).encode()
    with (tmp_path / "model.safetensors").open("wb") as file:
        file.write(struct.pack("<Q", len(serialized)) + serialized)
        file.truncate(8 + len(serialized) + total)

    assert discover(tmp_path) == "akas.models.inkling2.model:Inkling2Mini"
    checkpoint = Inkling2Mini.checkpoint_type(tmp_path, tokenizer=False)
    assert config.kv_bytes_per_slot(32) == 4 * (5 * 8 + 3 * 32) * 2 * 16
    graph = Inkling2Mini.compose(
        checkpoint, rows=2, chunk=4, max_requests=4, max_context=32, phase="prefill"
    ).graph
    assert graph.buffer_at("model.llm.layers.1.mlp.experts.w13_weight").shape == (
        4,
        64,
        32,
    )
    assert graph.buffer_at("model.llm.unembed.weight").shape == (60, 64)
    for pass_index in range(3):
        assert graph.buffer_at(f"layer.1.pass.{pass_index}.attn.keys").shape == (
            4,
            2,
            8,
            16,
        )
        assert graph.buffer_at(f"layer.2.pass.{pass_index}.attn.keys").shape == (
            4,
            2,
            32,
            16,
        )
    assert graph.buffer_at("logits").shape == (2 * 60,)
    assert sum(call.kind == "relative_prefill_attention" for call in graph.calls) == 8
    assert sum(call.kind == "block_scaled_experts" for call in graph.calls) == 16

    decode = Inkling2Mini.compose(
        checkpoint, rows=2, chunk=1, max_requests=4, max_context=32, phase="decode"
    ).graph
    assert sum(call.kind == "relative_decode_grouped" for call in decode.calls) == 8
