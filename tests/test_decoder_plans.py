"""Whole-plan contracts captured before extracting reusable decoder blocks.

The fixtures retain every operation's parameters, buffer identities, layouts,
labels and commit ordering. They were compared against the pre-refactor plans,
including both KV precisions and the dense draft/verify/fused-head paths.
"""

import importlib
import json
from pathlib import Path

import pytest

from plan_fixture import Buffer, record, dense_graph_contract, dense_frozen_contract


@pytest.mark.parametrize(
    "phase,options,fixture",
    [
        ("decode", {}, "decode-base"),
        ("decode", {"fp8": True}, "decode-fp8"),
        ("prefill", {}, "prefill-base"),
        ("prefill", {"fp8": True}, "prefill-fp8"),
        ("batch", {}, "batch-base"),
        ("batch", {"verify": True}, "batch-verify"),
        ("batch", {"draft": True}, "batch-draft"),
        ("batch", {"verify": True, "fused": True}, "batch-verify-fused"),
        ("batch", {"fused": True}, "batch-fused"),
    ],
)
def test_decoder_plan_preserves_complete_operation_contract(
    monkeypatch, phase, options, fixture
):
    if phase == "batch":
        expected = json.loads((Path(__file__).parent / "fixtures" / ("plan-" + fixture + ".json")).read_text())
        actual = dense_graph_contract(**options)
        assert actual == dense_frozen_contract(expected)
        return
    paths = {
        "decode": "akas.execution.bindings.block_fp8.decode",
        "prefill": "akas.execution.bindings.block_fp8.prefill",
    }
    module = importlib.import_module(paths[phase])
    import tensor.runtime.buffer

    monkeypatch.setattr(
        tensor.runtime.buffer,
        "Buffer",
        lambda device, pointer, shape, dtype, owner: Buffer(
            "view:" + owner.name, shape, dtype
        ),
    )
    actual = record(module, phase, **options)
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / ("plan-" + fixture + ".json")).read_text()
    )
    assert actual == expected
