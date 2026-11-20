"""Voice-enhancement graph, checkpoint and independent host expectations."""

import io
from pathlib import Path
import wave

import numpy as np
import pytest

from akas.compilation.compile import compile_forwards, plan
from akas.compilation.tuning import SearchSettings
from akas.models.voice_isolator.audio import float_wav, wav_samples
from akas.models.voice_isolator.config import VoiceIsolatorConfig
from akas.models.voice_isolator.weights import VoiceIsolatorCheckpoint


ROOT = Path(__file__).parents[1]
CHECKPOINT = ROOT / "build/checkpoints/voice-isolator-v1/denoiser.safetensors"


def test_forward_search_settings_accept_json_options(monkeypatch):
    captured = {}
    import akas.compilation.compile as module

    import akas.artifacts.read

    monkeypatch.setattr(
        akas.artifacts.read, "validate_forward_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        module,
        "build",
        lambda bundle, folder, **kwargs: captured.update(kwargs) or {"tuning": {}},
    )
    monkeypatch.setattr(module, "publish", lambda *args: None)
    monkeypatch.setattr(module, "checksum", lambda *args: "checksum")
    checkpoint = VoiceIsolatorCheckpoint.__new__(VoiceIsolatorCheckpoint)
    checkpoint.config = VoiceIsolatorConfig.from_metadata({})
    compile_forwards(
        checkpoint,
        ROOT / "build/unused-forward-options",
        definition=type(
            "Declaration",
            (),
            {
                "forward_index": ("speech.json", "schema"),
                "forward_phases": classmethod(
                    lambda cls, checkpoint, provider: (
                        {},
                        (
                            (
                                "enhance",
                                1,
                                __import__("akas.graph", fromlist=["Graph"]).Graph(
                                    contract="reference"
                                ),
                            ),
                        ),
                    )
                ),
            },
        ),
        provider="webgpu",
        device=object(),
        search_settings={"budget": 1},
    )
    assert isinstance(captured["search_settings"], SearchSettings)
    assert captured["search_settings"].budget == 1


def test_offline_config_and_exact_band_edges():
    config = VoiceIsolatorConfig.from_metadata({})
    assert config.width == 512
    assert config.chunk_samples == 384000
    assert config.frames(144000) == 301
    assert len(config.edges) == 46
    assert config.edges[0] == (0, 4)
    assert config.edges[-2:] == ((1023, 1024), (1024, 1025))
    assert sum(end - start for start, end in config.edges) == config.bins


@pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason="released Voice Isolator checkpoint not downloaded"
)
def test_checkpoint_inventory_and_webgpu_plan():
    checkpoint = VoiceIsolatorCheckpoint(CHECKPOINT)
    assert len(checkpoint.tensors) == 691
    assert sum(np.prod(info.shape) for info in checkpoint.tensors.values()) == 64047112
    from akas.models.voice_isolator.model import compose

    graph = compose(checkpoint, phase="enhance", rows=144000)
    assert len(graph.calls) == 705
    bundle = plan(graph, provider="webgpu")
    assert len(bundle.kernels) == 252
    kinds = {kernel.kind for kernel in bundle.kernels}
    assert {
        "stft",
        "istft_matrix",
        "istft_overlap",
        "full_attention",
        "linear_bias_activation",
        "selective_scan",
        "causal_depthwise_sequences",
        "complex_correction",
    } <= kinds
    assert not kinds & {"column_bias", "gelu", "softplus_bias"}


@pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason="released checkpoint not downloaded"
)
@pytest.mark.parametrize("samples", [144000, 384000])
@pytest.mark.parametrize("weight_dtype", ["float32", "float16"])
def test_forward_binds_every_lowered_kernel(samples, weight_dtype):
    pytest.importorskip("tilelang")
    from akas.compilation.validate import kernel_bindings
    from akas.graph import Graph
    from akas.kernels.definition import define
    from akas.models.voice_isolator.model import compose

    checkpoint = VoiceIsolatorCheckpoint(CHECKPOINT, weight_dtype=weight_dtype)
    bundle = plan(compose(checkpoint, phase="enhance", rows=samples), provider="webgpu")
    graph = Graph.from_metadata(bundle.manifest["forward"])
    assert any(entry.kind == "linear_w16a16" for entry in bundle.kernels) == (
        weight_dtype == "float16"
    )
    for entry in bundle.kernels:
        prim = define(*entry.source.args)
        signature = [
            dict(name=buf.name, shape=tuple(map(int, buf.shape)), dtype=str(buf.dtype))
            for buf in (prim.buffer_map[param] for param in prim.params)
        ]
        kernel_bindings(graph, entry.key, signature)


def test_position_and_projection_storage_policies():
    checkpoint = VoiceIsolatorCheckpoint.__new__(VoiceIsolatorCheckpoint)
    checkpoint.config = VoiceIsolatorConfig.from_metadata({})
    checkpoint.weight_dtype = "float16"
    checkpoint.fp32_projections = ()
    checkpoint.tensors = {
        "projection.weight": type("Info", (), {"shape": (4, 2)})(),
        "self_attn.in_proj_weight": type("Info", (), {"shape": (12, 4)})(),
        "recurrent.A_log": type("Info", (), {"shape": (4, 2)})(),
    }
    assert checkpoint.storage_dtype("projection.weight") == "float16"
    assert checkpoint.storage_dtype("self_attn.in_proj_weight") == "float16"
    assert checkpoint.storage_dtype("recurrent.A_log") == "float32"
    time = checkpoint.derived("position.blocks.0.time_transformer.attn@144000")
    band = checkpoint.derived("position.blocks.0.band_transformer.self_attn@144000")
    np.testing.assert_array_equal(time, np.tile(np.arange(301, dtype="int32"), 46))
    np.testing.assert_array_equal(band, np.tile(np.arange(46, dtype="int32"), 301))


@pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason="released checkpoint not downloaded"
)
@pytest.mark.parametrize(
    ("group", "selected", "count"),
    [
        ("mamba_coefficients", "blocks.1.time_mamba.forward_blocks.dt_proj.weight", 24),
        ("time_qkv", "blocks.0.time_transformer.attn.qkv.weight", 4),
        ("decoder_output", "denoise_decoder.heads.0.3.weight", 46),
    ],
)
def test_fp32_projection_groups(group, selected, count):
    from akas.models.voice_isolator.model import VoiceIsolator

    checkpoint = VoiceIsolatorCheckpoint(CHECKPOINT, weight_dtype="float16")
    half = [
        name
        for name in checkpoint.tensors
        if checkpoint.storage_dtype(name) == "float16"
    ]
    mixed = VoiceIsolatorCheckpoint(
        CHECKPOINT, weight_dtype="float16", fp32_projections=[group]
    )
    restored = {name for name in half if mixed.storage_dtype(name) == "float32"}
    assert len(restored) == count and selected in restored
    metadata, graphs = VoiceIsolator.forward_phases(
        mixed, provider="webgpu", chunks=(384000,)
    )
    assert metadata["fp32_projections"] == [group]
    _, _, graph = next(graphs)
    assert graph.buffers[selected].dtype == "float32"
    assert any(entry.kind == "linear_w16a16" for entry in graph.calls)


def test_pcm_io_preserves_unclipped_float_output():
    samples = np.linspace(-1.5, 1.5, 48, dtype="float32")
    raw = float_wav(samples)
    decoded, rate = wav_samples(raw)
    assert rate == 48000
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, samples)
    pcm = samples.clip(-1, 1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as source:
        source.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        source.writeframes((pcm * 32768).astype("<i2").tobytes())
    decoded, rate = wav_samples(buffer.getvalue())
    assert rate == 48000
    np.testing.assert_allclose(decoded, (pcm * 32768).astype("<i2") / 32768)


def test_long_window_placement_matches_upstream_rule():
    samples, chunk, overlap = 11 * 48000, 8 * 48000, 2 * 48000
    starts = list(range(0, samples - chunk, chunk - overlap)) + [samples - chunk]
    assert starts == [0, 3 * 48000]
    assert starts[-1] + chunk == samples


def test_golden_hashes_are_pinned():
    for suffix, expected in (
        ("", "ba34cb451867fd3bb5b2c4e18f314d22e8379556d7a9dabb73bbf8ac2c45443b"),
        ("_11s", "f54d1971b0e6d5daa35660b274a712e26c75fb37cc4870ceb8082a7f331c4bf1"),
    ):
        from hashlib import sha256

        path = ROOT / f"build/datasets/voice-isolator-golden/input{suffix}.wav"
        if path.is_file():
            assert sha256(path.read_bytes()).hexdigest() == expected
