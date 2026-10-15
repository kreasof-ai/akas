"""Uniform codec contracts and executable conversion/fused-consumer qualification."""

import ast
from contextlib import ExitStack
from importlib import import_module
import os
from pathlib import Path

import numpy as np
import pytest

from akas.quantization import get

CODECS = [
    ("activations", "fp8"),
    ("activations", "q8"),
    ("activations", "q16"),
    ("weights", "fp8"),
    ("weights", "ggml"),
    ("kv", "fp8"),
]


def cases():
    for name in ("q8", "q16"):
        yield f"quantize_{name}", dict(r=3, k=256), "reference"
        yield f"dequantize_{name}", dict(r=3, k=256), "reference"
    yield "quantize_q16", dict(r=3, k=256, fixed_residual=True), "reference"
    yield "quantize", dict(r=3, k=256), "block-fp8"
    yield "quantize", dict(r=2, top=3, k=128), "block-fp8"
    for kind in ("dequantize_fp8", "dequantize_weight_fp8"):
        yield kind, dict(r=3, k=256), "reference"
        yield kind, dict(shape=(2, 3, 128)), "reference"
    for encoding in (0, 1, 2, 3, 8, 12, 13, 14):
        yield "dequantize_ggml", dict(r=3, k=256, type=encoding), "reference"
    for kind in ("kv_encode", "kv_decode"):
        yield kind, dict(slots=2, chunk=2, kv_heads=3, d=256, capacity=7), "kv-pair"
    for kind in ("kv_encode", "kv_decode"):
        yield kind, dict(slots=2, chunk=2, kv_heads=1, d=64, capacity=7), "kv-pair"
    for rows in (1, 3):
        for activation in (None, "q8", "q16", "fp8"):
            for weight in ("ggml", "fp8"):
                yield (
                    "quantized_linear",
                    dict(
                        r=rows,
                        k=256,
                        o=33,
                        activation_format=activation,
                        weight_format=weight,
                        type=2,
                    ),
                    "quantized",
                )
    yield (
        "quantized_linear",
        dict(r=3, k=96, o=7, activation_format="q16", type=8),
        "quantized",
    )


CASES = list(cases())


@pytest.mark.parametrize("scope,name", CODECS)
@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_every_format_binds_named_storage_and_inline_access(scope, name, provider):
    format = get(scope, name)
    shape = (2, 3, 7, 256) if scope == "kv" else (3, 256)
    options = {"encoding": 2} if name == "ggml" else {}
    binding = format.bind(shape, provider=provider, **options)
    assert tuple(binding.storage.named.values()) == binding.storage.buffers
    assert binding.storage == format.storage(shape, provider=provider, **options)
    for capability in ("read", "read_tile"):
        assert capability in format.capabilities(provider)
        assert callable(binding.require(capability))
    if name in ("q8", "q16"):
        assert binding.storage.metadata_names == ("scales", "sums")
        assert binding.read_buffers == (
            binding.storage.values,
            binding.storage.named["scales"],
        )
    with pytest.raises(ValueError, match="unknown inline codec capability"):
        binding.require("missing")


def test_unsupported_geometry_fails_before_kernel_generation():
    for provider in ("cuda", "webgpu"):
        for shape in ((1, 33), (32, 1)):
            with pytest.raises(ValueError, match="whole blocks"):
                get("weights", "ggml").storage(shape, provider=provider, encoding=2)
    with pytest.raises(ValueError, match="whole blocks"):
        get("weights", "ggml").storage((2, 128), provider="cuda", encoding=12)


def test_metadata_must_be_named_at_declaration():
    from akas.memory.state import BufferSpec
    from akas.quantization import Storage

    value = BufferSpec((32,), "uint8")
    scale = BufferSpec((1,), "float32")
    with pytest.raises(ValueError, match="uniquely name every metadata buffer"):
        Storage(value, (scale,))
    with pytest.raises(ValueError, match="uniquely name every metadata buffer"):
        Storage(value, (scale,), ("values",))


@pytest.mark.parametrize("rows,depth", [(3, 256), (1, 96)])
def test_legacy_packed_q8_rejects_unsupported_geometry(rows, depth):
    from akas.kernels.registry import select
    from akas.kernels.definition import KernelSpec
    from akas.ops.linear.webgpu import linear_q8

    p = dict(r=rows, k=depth, o=8, type=2)
    with pytest.raises(ValueError, match="no compatible implementation"):
        select("linear_q8", p, contract="packed", provider="webgpu")
    with pytest.raises(ValueError, match="packed Q8 dot requires"):
        linear_q8(KernelSpec("linear_q8", p), {})


def test_word_packing_preserves_odd_block_count_and_padding():
    from akas.quantization._shared.packing import pack_words
    from akas.quantization.weights.ggml.validation import sample_weight

    storage = get("weights", "ggml").storage((3, 256), provider="webgpu", encoding=14)
    raw = np.arange(630, dtype="uint16").astype("uint8")
    packed = pack_words(raw)
    assert packed.shape == storage.values.shape == (158,)
    np.testing.assert_array_equal(packed.view("uint8")[:630], raw)
    np.testing.assert_array_equal(packed.view("uint8")[630:], 0)
    assert sample_weight(packed, 14).shape == (3 * 256,)


def test_fp8_preparation_preserves_payload_and_bf16_scales_across_providers():
    from akas.quantization.weights.fp8.kernel import prepare_host

    bits = np.arange(3 * 256, dtype="uint16").reshape(3, 256).astype("uint8") % 127
    scales = np.array([[0.12345, 0.4321]], dtype="float32")
    cuda = prepare_host(bits, scales, provider="cuda")
    webgpu = prepare_host(bits, scales, provider="webgpu")
    np.testing.assert_array_equal(cuda["values"], webgpu["values"].view("uint8"))
    np.testing.assert_array_equal(cuda["scales"], webgpu["scales"])
    assert np.any(cuda["scales"] != scales)


def test_kv_persistent_layout_keeps_provider_packing():
    format = get("kv", "fp8")
    shape = (2, 3, 7, 64)
    from akas.quantization.kv import pair_layout

    cuda, webgpu = (pair_layout("fp8", shape, provider=p) for p in ("cuda", "webgpu"))
    assert cuda[0].shape == shape and cuda[0].dtype == "uint8"
    assert webgpu[0].shape == (2, 3, 7, 16) and webgpu[0].dtype == "uint32"
    assert cuda[2:] == webgpu[2:]
    binding = format.bind(shape, provider="cuda")
    assert binding.write_pair is None
    assert callable(binding.require("write"))
    with pytest.raises(ValueError, match="does not support write_pair"):
        binding.require("write_pair")


def test_format_lookup_contributes_dynamic_codec_sources_to_artifacts():
    from akas.compilation.emit import source_dependencies

    dependencies = source_dependencies("akas.ops.vocabulary.kernel")
    for scope, name in CODECS:
        assert f"akas.quantization.{scope}.{name}.definition" in dependencies
        for provider in ("cuda", "webgpu"):
            assert f"akas.quantization.{scope}.{name}.{provider}" in dependencies


def test_codecs_do_not_depend_on_checkpoint_parsing_or_operation_kernels():
    root = Path(__file__).parents[1] / "src/akas/quantization"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("akas.checkpoints", "akas.ops")
                ), path
    for scope, name in CODECS:
        package = import_module(f"akas.quantization.{scope}.{name}")
        assert package.__all__ == ["FORMAT"]


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("kind,parameters,contract", CASES)
def test_factories_return_explicit_results_on_both_providers(
    provider, kind, parameters, contract
):
    from akas.kernels.definition import KernelSpec, Capabilities, define

    result = define(
        KernelSpec(kind, parameters, contract), capabilities=Capabilities(provider)
    )
    assert result.__akas_signature__.outputs
    assert not result.__akas_signature__.mutates or kind == "kv_encode"


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("kind,parameters,contract", CASES)
def test_native_codec_and_fused_consumer_qualification(
    tmp_path, provider, kind, parameters, contract
):
    enabled = (
        os.environ.get("TENSOR_QWEN_CUDA") == "1"
        if provider == "cuda"
        else os.environ.get("AKAS_COMPILE_WEBGPU") == "1"
    )
    if not enabled:
        pytest.skip("opt-in CUDA execution or WebGPU compilation")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import validation

    entry = tmp_path / "kernel.py"
    artifact = tmp_path / "kernel.tbin"
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    entry.write_text(
        source(
            kind, parameters, contract=contract, provider=provider, target=target
        ).render()
    )
    build_artifact(entry, artifact, provider=provider, target=target)
    meta, files = read_artifact(artifact)
    if provider == "webgpu":
        assert any(name.endswith(".wgsl") for name in files)
        return
    case = validation(kind, parameters, meta["arguments"])
    with tensor.Device() as device, ExitStack() as stack:
        buffers = []
        for argument in meta["arguments"]:
            buffer = device.from_numpy(
                case.arguments[argument["name"]], dtype=argument["dtype"]
            )
            stack.callback(buffer.release)
            buffers.append(buffer)
        kernel = stack.enter_context(device.load(artifact))
        kernel.launch(*buffers)
        for argument, buffer in zip(meta["arguments"], buffers):
            name = argument["name"]
            actual, expected = buffer.to_numpy(), case.expected[name]
            if name in case.comparators:
                case.comparators[name](actual, expected)
            elif np.issubdtype(actual.dtype, np.integer):
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(
                    actual, expected, rtol=case.rtol, atol=case.atol
                )
