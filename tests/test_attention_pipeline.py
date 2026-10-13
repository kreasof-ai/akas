"""Split-context policies lower into qualified operation stages on the common runtime."""

import os
import pytest

from akas.graph import Graph
from akas.ops.attention.lowering import lower


def attention_graph(slots=2, chunk=2, capacity=128):
    graph = Graph(contract="block-fp8", phase="prefill")
    specs = {
        "q": ((slots * chunk, 16, 256), "bfloat16"),
        "kc": ((slots, 2, capacity, 256), "uint8"),
        "vc": ((slots, 2, capacity, 256), "uint8"),
        "ks": ((slots, 2, capacity, 2), "float32"),
        "vs": ((slots, 2, capacity, 2), "float32"),
        "projection": ((slots * chunk, 8192), "float32"),
        "positions": ((slots,), "int32"),
        "lengths": ((slots,), "int32"),
        "out": ((slots * chunk, 4096), "bfloat16"),
    }
    buffers = {name: graph.buffer(name, shape, dtype, role="output" if name == "out" else "input")
               for name, (shape, dtype) in specs.items()}
    graph("attention", dict(slots=slots, chunk=chunk, capacity=capacity, kv_dtype="fp8"), **buffers)
    return graph


def test_split_attention_is_compiler_independent_and_reuses_original_output():
    graph, commit = lower(attention_graph(), None, dict(splits=4), provider="cuda", target="sm_89")
    assert commit is None
    assert [call.kind for call in graph.calls] == ["attention_chunk_partial", "attention_chunk_active", "attention_chunk_merge"]
    assert graph.calls[-1].arguments[-1] == "out"
    assert graph.buffers[graph.calls[0].arguments[-2]].shape == (2, 2, 4, 16, 256)


def test_decoded_workspace_is_scratch_and_requires_compatible_target():
    settings = dict(splits=4, hopper=True, decoded_kv=True)
    graph, _ = lower(attention_graph(), None, settings, provider="cuda", target="sm_90a")
    assert graph.calls[0].kind == "attention_decode_workspace"
    assert all(graph.buffers[name].role == "scratch" for name in graph.calls[0].arguments[-2:])
    with pytest.raises(ValueError, match="sm_90a"):
        lower(attention_graph(), None, settings, provider="cuda", target="sm_89")


def test_custom_hardware_factory_serialization_preserves_routing_and_arguments(monkeypatch):
    import json
    from types import SimpleNamespace
    from akas.ops.contracts import Implementation
    from akas.compilation.selection import source
    choice = Implementation("attention_chunk_partial", "attention.hardware.hopper.attention", "custom",
                            factory="partial", parameters_only=True, targets=("sm_90a",))
    declaration = source("attention_chunk_partial", dict(slots=2, chunk=8, capacity=128),
                         contract="custom", target="sm_90a", catalog=(choice,))
    text = declaration.render()
    assert "def tensor_export" in text and "akas.ops.definition import define" in text
    observed = []
    monkeypatch.setattr("akas.ops.definition.import_module", lambda name: SimpleNamespace(
        partial=lambda p: observed.append((name, p)) or "kernel"))
    namespace = {}
    exec(text, namespace)
    assert namespace["tensor_export"]() == {"kernel": "kernel"}
    assert observed == [(choice.module_name, dict(slots=2, chunk=8, capacity=128))]
    # Routing survives the JSON representation used for source metadata.
    assert json.loads(json.dumps(declaration.args))[3]["factory"] == "partial"


@pytest.mark.skipif(os.environ.get("AKAS_FORWARD_CUDA") != "1", reason="native attention pipeline qualification")
def test_split_pipeline_default_beam_qualifies_all_physical_stages(tmp_path):
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    graph, _ = lower(attention_graph(), None, dict(splits=2), provider="cuda", target="sm_89")
    result = produce(graph, tmp_path, search_settings=SearchSettings(budget=1, repeats=1, samples=1))
    assert len(result["tuning"]["operations"]) == 3
    assert all(operation["selected"]["score"] > 0 for operation in result["tuning"]["operations"])
