"""Program policies reuse operation-owned optimization paths across model phases."""

import pytest
from akas.compilation.policy import lower


def test_empty_policy_remains_lazy_for_new_model_contract(monkeypatch):
    from akas.graph import Graph
    import akas.ops.registry

    graph = Graph(contract="new-model")
    monkeypatch.setattr(
        akas.ops.registry,
        "planning_rules",
        lambda: pytest.fail("empty policy imported operations"),
    )
    assert lower(graph) == (graph, None)


@pytest.mark.parametrize("phase", [None, "verify", "proposal"])
def test_projection_routing_recompute_and_hopper_policies_share_program_graphs(phase):
    from test_moe_speculative_program import checkpoint
    from akas.models.qwen35.model import compose_moe_program
    from akas.compilation.compile import plan
    from akas.artifacts.schema import Source

    loaded = checkpoint()
    composed = compose_moe_program(
        loaded,
        rows=2,
        chunk=4,
        max_requests=4,
        max_context=128,
        phase=phase,
        kv_dtype="fp8",
    )
    graph, commit = composed if isinstance(composed, tuple) else (composed, None)
    policies = {
        "linear": {
            "hopper": True,
            "mma_reduction": 32,
            "mma_reorder": True,
            "columns": 64,
        },
        "moe": {
            "hopper": True,
            "compact": True,
            "bf16_mma": True,
            "mma_reduction": 32,
            "mma_reorder": True,
        },
        "attention": {
            "hopper": True,
            "decoded_kv": True,
            "splits": 2,
            "query_tokens": 4,
        },
        "gdn": {"commit": "recompute"},
    }
    graph, commit = lower(graph, commit, policy=policies, target="sm_90a")
    bundle = plan(graph, target="sm_90a")
    assert {"projection_partial", "expert_partial", "attention_chunk_partial"} <= {
        call.kind for call in graph.calls
    }
    for declaration in bundle.kernels:
        source = declaration.source
        if source.module == "akas.ops.state.mapping":
            source = Source(source.args[0], source.args[1], source.args[2])
        assert source.module == "akas.ops.definition" and source.factory == "define"
    if phase == "verify":
        assert [call.kind for call in commit.calls] == [
            "gdn_conv",
            "gdn_scan",
            "advance",
        ]
        plan(commit, target="sm_90a")


def test_operation_policy_rejects_unknown_keys_and_incompatible_hardware():
    from akas.graph import Graph

    graph = Graph(contract="block-fp8")
    with pytest.raises(ValueError, match="unknown operation"):
        lower(graph, policy={"forgotten-model": {}})
    with pytest.raises(ValueError, match="unknown projection"):
        lower(graph, policy={"linear": {"unknown": True}})
    with pytest.raises(ValueError, match="sm_90a"):
        lower(graph, policy={"linear": {"hopper": True}}, target="sm_89")
