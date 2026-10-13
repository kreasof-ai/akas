"""Operation write metadata is checked against independent generated kernel IR."""

import pytest

# Small representative ABIs exercise metadata without compiling GPU artifacts.
CASES = [
    ("reference", "linear", dict(r=1, k=64, o=16)),
    ("reference", "embedding", dict(r=1, c=64, v=16)),
    ("reference", "argmax", dict(r=1, c=64)),
    ("reference", "swiglu", dict(r=1, c=64)),
    ("reference", "add", dict(r=1, c=64)),
    ("reference", "rms", dict(r=1, c=64, eps=1e-6)),
    ("reference", "add_rms", dict(r=1, c=64, eps=1e-6)),
    ("reference", "rope", dict(r=1, h=2, d=64, theta=10000)),
    ("reference", "conv", dict(r=1, chunk=2, c=64)),
    ("reference", "chunk_controls", dict(r=2, chunk=4)),
    ("reference", "chunk_last", dict(r=2, chunk=4, c=64)),
    ("reference", "masked_copy", dict(r=2, c=64)),
    ("reference", "chunk_tail", dict(r=2, chunk=4, tail=2, c=64)),
    ("reference", "tail_controls", dict(r=2, chunk=4, tail=2)),
    ("bf16", "add_rms", dict(r=1, c=128, eps=1e-6)),
    ("bf16", "gdn_conv", dict(r=2, slots=1, chunk=2, pool=2, channels=128, projection=128)),
    ("bf16", "gdn_conv", dict(r=2, slots=1, chunk=2, pool=2, channels=128, projection=128, commit=False)),
    ("bf16", "gdn_scan", dict(r=2, slots=1, chunk=2, pool=2, heads=1, d=32)),
    ("bf16", "gdn_scan", dict(r=2, slots=1, chunk=2, pool=2, heads=1, d=32, commit=False)),
]


CASES = [(*case, "forward") for case in CASES] + [
    ("block-fp8", "gdn_conv", dict(slots=1, chunk=2), "verification"),
    ("block-fp8", "gdn_conv", dict(slots=1, chunk=2, full_checkpoints=True, commit=False), "verification"),
    ("block-fp8", "gdn_scan", dict(slots=1, chunk=2), "verification"),
    ("block-fp8", "gdn_scan", dict(slots=1, chunk=2, full_checkpoints=True, commit=False), "verification"),
]

@pytest.mark.parametrize("contract,kind,parameters,phase", CASES)
def test_declared_outputs_match_actual_primitive_writes(contract, kind, parameters, phase):
    pytest.importorskip("tilelang")
    from tvm import tirx as ir
    from akas.graph import Call
    from akas.ops.registry import graph_outputs, select

    variant = select(kind, parameters, contract=contract, provider="cuda", phase=phase, target="sm_89")
    kernel = variant.make(parameters)
    arguments = tuple(kernel.buffer_map[parameter].name for parameter in kernel.params)
    buffers = {buffer.data: buffer.name for buffer in kernel.buffer_map.values()}
    written = set()
    def visit(node):
        if isinstance(node, ir.BufferStore) and node.buffer.data in buffers:
            written.add(buffers[node.buffer.data])
        elif isinstance(node, ir.Call):
            name = getattr(node.op, "name", "")
            if name == "tl.region" and isinstance(node.args[0], ir.BufferLoad):
                load, access = node.args[0], node.args[1]
                if hasattr(access, "value") and int(access.value) & 2 and load.buffer.data in buffers:
                    written.add(buffers[load.buffer.data])
            elif name in ("tirx.tvm_access_ptr", "tir.tvm_access_ptr"):
                access = node.args[4]
                if hasattr(access, "value") and int(access.value) & 2 and node.args[1] in buffers:
                    written.add(buffers[node.args[1]])
    ir.stmt_functor.post_order_visit(kernel.body, visit)
    ordered = graph_outputs(Call(kind, parameters, arguments), contract=contract)
    assert ordered is not None
    assert set(ordered) == written
    named = graph_outputs(Call(kind, parameters, bindings=dict(zip(arguments, arguments))), contract=contract)
    if named is not None:
        assert set(named) == written
