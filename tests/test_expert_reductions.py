"""Compact reduction partitions reuse the native operation qualification."""
import os
import pytest
from test_lfm_moe_native import test_native_operations_match_independent_references as check_operation


@pytest.mark.skipif(os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="native CUDA reduction qualification")
@pytest.mark.parametrize("kind,parameters", [
    ("expert_partial", dict(r=1, rows=1, k=384, o=256, experts=7, top=3,
        partitions=3, block_m=16, compact=True, tile_map=True, columns=128,
        threads=128, stages=1, activation_vector=16, routed_input=True, mma_dtype="float16")),
    ("expert_partial", dict(r=4, rows=4, k=768, o=256, experts=7, top=3,
        partitions=3, block_m=16, compact=True, tile_map=True, columns=64,
        threads=128, stages=2, activation_vector=16)),
    ("expert_merge", dict(r=3, top=4, o=256, partitions=3, output_dtype="bfloat16")),
    ("expert_merge", dict(r=3, top=4, o=256, partitions=3, output_dtype="float16")),
    ("expert_merge", dict(r=3, top=4, o=256, partitions=3)),
    ("expert_pair_partial", dict(r=1, rows=1, k=512, o=256, experts=7, top=3,
        partitions=4, block_m=16, compact=True, tile_map=True, columns=128,
        threads=128, stages=1, activation_vector=16, mma_dtype="float16", pair_layout="wide")),
    ("expert_pair_partial", dict(r=4, rows=4, k=768, o=256, experts=7, top=3,
        partitions=3, block_m=16, compact=True, tile_map=True, columns=64,
        threads=128, stages=2, activation_vector=16, pair_layout="separate")),
    ("swiglu_experts", dict(r=4, top=3, c=513, partitions=3, input_dtype="float32",
        output_dtype="bfloat16", round_inputs=True, round_dtype="bfloat16")),
    ("swiglu_experts", dict(r=1, top=3, c=512, partitions=4, input_dtype="float32",
        output_dtype="float16", round_inputs=True, round_dtype="float16")),
])
def test_split_expert_operations_match_independent_equations(tmp_path, kind, parameters):
    check_operation(tmp_path, kind, parameters)


@pytest.mark.parametrize("rows,chunk,split", [(1, 1, True), (1, 4, True), (2, 4, False)])
def test_partition_policy_keeps_partial_accumulation_and_final_storage_separate(tmp_path, rows, chunk, split):
    from test_lfm_moe_native import fixture
    from akas.models.lfm2 import LFM2MoE
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    checkpoint = LFM2MoE.checkpoint_type(fixture(tmp_path / "weights"), tokenizer=False)
    graph = LFM2MoE.compose(checkpoint, rows=rows, chunk=chunk, max_requests=4, max_context=64).graph
    graph, _ = lower(graph, policy={"moe": dict(fused_swiglu=True, partitions=2,
        partition_maximum_rows=4, intermediate_dtype="bfloat16", activation_dtype="bfloat16")})
    projections = [c for c in graph.calls if c.kind == "expert_partial"]
    merges = [c for c in graph.calls if c.kind == "expert_merge"]
    assert len(merges) == int(split)
    assert projections and projections[0].parameters["partitions"] == (2 if split else 1)
    output = projections[0].bindings["out"]
    assert graph.buffers[output].dtype == ("float32" if split else "bfloat16")
    if split:
        assert graph.buffers[output].shape[-2] == 2
        assert merges[0].parameters["output_dtype"] == "bfloat16"
        assert graph.buffers[merges[0].bindings["out"]].dtype == "bfloat16"
    plan(graph, target="sm_89")


@pytest.mark.parametrize("parts,maximum", [(0, 4), (-1, 4), (True, 4), (1.5, 4), (2, 0), (2, True)])
def test_partition_policy_rejects_invalid_limits(parts, maximum):
    from akas.compilation.policy import lower
    from akas.graph import Graph

    with pytest.raises(ValueError, match="partitions|maximum rows"):
        lower(Graph(contract="reference"), policy={"moe": dict(partitions=parts, partition_maximum_rows=maximum)})
