"""Weight formats do not choose request-state addressing."""

import os
import numpy as np
import pytest
from pathlib import Path
from support.migration import source as export_kernel
from akas.compilation.emit import build_artifact


def test_state_mapping_rejects_a_copy_spanning_multiple_requests():
    import tilelang.language as T
    from akas.compilation.transforms.state import map_state

    @T.prim_func
    def kernel(state: T.Tensor((2, 4), "float32"), out: T.Tensor((2, 4), "float32")):
        with T.Kernel(1, threads=64):
            T.copy(state, out)

    with pytest.raises(ValueError, match="crosses request rows"):
        map_state(kernel, (0,), rows=2, capacity=4)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_fp8_cache_state_is_written_to_reordered_request_slots(tmp_path, provider):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable hardware state-pool qualification")
    import tensor

    r, pool, heads, d, cap = 2, 4, 3, 64, 8
    p = dict(
        r=r, h=heads, kh=heads, d=d, capacity=cap, kv_dtype="fp8", provider=provider
    )
    source = tmp_path / "mapped.py"
    source.write_text(
        export_kernel(
            "akas.compilation.transforms.state",
            "make_kernel",
            "akas.ops.attention.reference",
            "make_kernel",
            ("cache_write", p),
            (2, 3, 4, 5),
            r,
            pool,
        )
    )
    rng = np.random.default_rng(10)
    arrays = [rng.normal(size=(r, heads * d)).astype("float32") for _ in range(2)]
    arrays += [np.array([2, 5], dtype="int32"), np.ones(r, dtype="int32")]
    with tensor.Device(provider=provider) as device:
        artifact = tmp_path / "mapped.tbin"
        build_artifact(source, artifact, target=device.info["arch"], provider=provider)
        # FP8 uses bytes on CUDA and packed u32 words on portable WebGPU.
        packed = provider == "webgpu"
        shape = (pool, heads, cap, d // 4 if packed else d)
        dtype = "uint32" if packed else "uint8"
        state = [device.from_numpy(np.zeros(shape, dtype=dtype)) for _ in range(2)]
        scales = [
            device.from_numpy(np.zeros((pool, heads, cap, 1), dtype="float32"))
            for _ in range(2)
        ]
        inputs = [device.from_numpy(a) for a in arrays]
        mapping = device.from_numpy(np.array([3, 1], dtype="int32"))
        try:
            with device.load(artifact) as kernel:
                kernel.launch(*inputs[:2], *state, *scales, *inputs[2:], mapping)
                device.synchronize()
            for buffer in [*state, *scales]:
                result = buffer.to_numpy()
                np.testing.assert_array_equal(result[[0, 2]], 0)
                assert np.any(result[3, :, 2]) and np.any(result[1, :, 5])
                assert not np.any(result[3, :, :2])
                assert not np.any(result[1, :, :5])
        finally:
            for buffer in [*inputs, *state, *scales, mapping]:
                buffer.release()


@pytest.mark.parametrize("kind", ["gdn_recurrent", "attention"])
def test_native_state_algorithms_preserve_math_with_reordered_pool(tmp_path, kind):
    if os.environ.get("TENSOR_QWEN_CUDA") != "1":
        pytest.skip("enable native request mapping qualification")
    import tensor
    from support.migration import source as export_source
    from tensor.runtime.dtypes import decode_bfloat16
    from tensor.runtime.dtypes import encode_bfloat16

    rows, pool = 2, 4
    if kind == "gdn_recurrent":
        module = "akas.ops.gdn.cuda.decode"
        p = dict(slots=rows, heads=2, key=32, value=32, value_tile=16)
        states = (6,)
    else:
        module = "akas.ops.attention.cuda"
        p = dict(slots=rows, chunk=2, capacity=64)
        states = (1, 2)
    paths = []
    for index in range(2):
        path = tmp_path / f"kernel-{index}.py"
        text = (
            export_source(module, "make_kernel", kind, p, dependencies=())
            if index == 0
            else export_kernel(
                "akas.compilation.transforms.state",
                "make_kernel",
                module,
                "make_kernel",
                (kind, p),
                states,
                rows,
                pool,
            )
        )
        path.write_text(text)
        image = path.with_suffix(".tbin")
        build_artifact(path, image, target="sm_89")
        paths.append(image)
    rng = np.random.default_rng(34)
    mapping = np.array([3, 1], dtype="int32")
    with tensor.Device() as device:
        baseline, mapped = [device.load(path) for path in paths]
        initial, expected_pool, inputs, full_inputs = {}, {}, [], []
        for index, spec in enumerate(baseline.manifest["arguments"]):
            shape, dtype = tuple(spec["shape"]), spec["dtype"]
            full_shape = (pool, *shape[1:]) if index in states else shape
            if dtype == "int32":
                host = np.full(
                    full_shape,
                    2
                    if spec["name"] == "lengths"
                    else 3
                    if spec["name"] == "positions"
                    else 1,
                    dtype="int32",
                )
            else:
                host = rng.normal(0, 0.05, full_shape).astype("float32")
                if dtype == "bfloat16":
                    host = decode_bfloat16(encode_bfloat16(host))
            if index in states:
                expected_pool[index] = host.copy()
                small = host[mapping].copy()
            else:
                small = host.copy()
            inputs.append(
                device.from_numpy(small, dtype="bfloat16")
                if dtype == "bfloat16"
                else device.from_numpy(small)
            )
            full_inputs.append(
                device.from_numpy(host, dtype="bfloat16")
                if dtype == "bfloat16"
                else device.from_numpy(host)
            )
        request_map = device.from_numpy(mapping)
        baseline.launch(*inputs)
        mapped.launch(*full_inputs, request_map)
        np.testing.assert_array_equal(inputs[-1].to_numpy(), full_inputs[-1].to_numpy())
        for index in states:
            expected_pool[index][mapping] = inputs[index].to_numpy()
            np.testing.assert_array_equal(
                full_inputs[index].to_numpy(), expected_pool[index]
            )


@pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1",
    reason="native mapped state alias regression",
)
@pytest.mark.parametrize("implementation", ["view", "nd"])
def test_mapped_restore_flattens_view_indices_using_view_shape(
    tmp_path, implementation
):
    import numpy as np
    import tensor
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward

    graph = Graph(contract="block-fp8", phase="commit")
    saved = graph.buffer("saved", (2, 4, 3, 4), "float32", role="input")
    state = graph.buffer("state", (4, 3, 4), "float32", role="state")
    accepted = graph.buffer("accepted", (2,), "int32", role="input")
    mapping = graph.buffer("mapping", (2,), "int32", role="input")
    graph(
        "restore",
        dict(slots=2, chunk=4, shape=[3, 4], full_checkpoints=True),
        checkpoints=saved,
        state=state,
        accepted=accepted,
        lengths=accepted,
    )
    from akas.kernels.contracts import Implementation

    catalog = (
        (
            Implementation(
                "restore", "gdn.cuda.verification", "block-fp8", factory="kernel"
            ),
        )
        if implementation == "view"
        else None
    )
    produce(graph, tmp_path, search=False, catalog=catalog)
    checkpoints = np.arange(96, dtype="float32").reshape(2, 4, 3, 4)
    initial = np.arange(48, dtype="float32").reshape(4, 3, 4) * -1 - 1
    expected = initial.copy()
    expected[3] = checkpoints[0, 1]
    with tensor.Device() as device:
        buffers = {}
        try:
            for name, value in (
                ("saved", checkpoints),
                ("state", initial),
                ("accepted", np.array([2, 0], dtype="int32")),
                ("mapping", np.array([3, 1], dtype="int32")),
            ):
                buffers[name] = device.from_numpy(value)
            with Forward(tmp_path, device, buffers) as forward:
                forward.run()
                np.testing.assert_array_equal(buffers["state"].to_numpy(), expected)
        finally:
            for value in buffers.values():
                value.release()


@pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1",
    reason="native inactive request isolation",
)
def test_inactive_collective_state_writes_cannot_overwrite_live_slot_zero(tmp_path):
    import numpy as np
    import tensor

    source = tmp_path / "mapped.py"
    source.write_text("""import tilelang.language as T
from akas.compilation.transforms.state import map_state

def tensor_export():
    @T.prim_func
    def original(state:T.Tensor((2,64),"float32"), active:T.Tensor((2,),"int32")):
        with T.Kernel(2,threads=64) as row:
            local = T.alloc_fragment((64,),"float32")
            T.copy(state[row,:],local)
            for column in T.Parallel(64):
                local[column] += T.cast(active[row],"float32")
            T.copy(local,state[row,:])
    return {"kernel":map_state(original,(0,),rows=2,capacity=4)}
""")
    artifact = tmp_path / "mapped.tbin"
    tensor.build(source, artifact, target="sm_89")
    with tensor.Device() as device:
        state = device.zeros((4, 64), "float32")
        active = device.from_numpy(np.array([1, 0], dtype="int32"))
        mapping = device.from_numpy(np.array([0, -1], dtype="int32"))
        kernel = device.load(artifact)
        try:
            for _ in range(100):
                kernel.launch(state, active, mapping)
            expected = np.zeros((4, 64), dtype="float32")
            expected[0] = 100
            np.testing.assert_array_equal(state.to_numpy(), expected)
        finally:
            kernel.release()
            mapping.release()
            active.release()
            state.release()


@pytest.mark.parametrize("capacity", [1, 4])
def test_read_only_state_keeps_mapping_after_simplification(capacity):
    from support.migration import factory as _historical_factory
    from akas.compilation.transforms.state import map_state

    original = _historical_factory(
        "akas.ops.attention.reference",
        "make_kernel",
        "rope",
        dict(r=1, h=1, d=64, theta=10000.0),
    )
    import tvm
    from tvm import tirx as ir

    mapped = map_state(original, (2,), rows=1, capacity=capacity)
    simplified = ir.transform.Simplify()(tvm.IRModule({"main": mapped}))["main"]
    reads = []
    ir.stmt_functor.post_order_visit(
        simplified.body,
        lambda node: (
            reads.append(node.buffer.name) if isinstance(node, ir.BufferLoad) else None
        ),
    )
    assert "mapping" in reads


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("capacity", [1, 4])
def test_read_only_state_compiles_and_uses_mapped_position(
    tmp_path, provider, capacity
):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable hardware state-pool qualification")
    import tensor

    source = tmp_path / "mapped_rope.py"
    source.write_text(
        export_kernel(
            "akas.compilation.transforms.state",
            "make_kernel",
            "akas.ops.attention.reference",
            "make_kernel",
            ("rope", dict(r=1, h=1, d=64, theta=10000.0)),
            (2,),
            1,
            capacity,
        )
    )
    rng = np.random.default_rng(512)
    x = rng.normal(size=64).astype("float32")
    slot = capacity - 1
    positions = np.zeros(capacity, dtype="int32")
    positions[slot] = 3
    angle = 3 * np.exp(-np.log(10000.0) * 2 / 64 * np.arange(32))
    expected = np.concatenate(
        (
            x[:32] * np.cos(angle) - x[32:] * np.sin(angle),
            x[:32] * np.sin(angle) + x[32:] * np.cos(angle),
        )
    )
    with tensor.Device(provider=provider) as device:
        artifact = source.with_suffix(".tbin")
        build_artifact(source, artifact, target=device.info["arch"], provider=provider)
        with device.load(artifact) as kernel:
            assert "mapping" in {arg["name"] for arg in kernel.manifest["arguments"]}
            buffers = [
                device.from_numpy(x),
                device.zeros((1, 64), "float32"),
                device.from_numpy(positions),
                device.from_numpy(np.array([slot], dtype="int32")),
            ]
            try:
                kernel.launch(*buffers)
                np.testing.assert_allclose(
                    buffers[1].to_numpy().ravel(), expected, rtol=2e-5, atol=2e-5
                )
                if capacity == 1:
                    from akas.memory.transfer import write

                    previous = buffers[1].to_numpy()
                    write(device, buffers[3], np.array([-1], dtype="int32"))
                    kernel.launch(*buffers)
                    np.testing.assert_array_equal(buffers[1].to_numpy(), previous)
            finally:
                for buffer in buffers:
                    buffer.release()


def test_state_mapping_preserves_reinterpreted_storage_dtype():
    import tilelang.language as T
    from tvm import tirx as ir
    from akas.compilation.transforms.state import map_state

    @T.prim_func
    def kernel(state: T.Tensor((2, 4), "uint8"), out: T.Tensor((2, 4), "float32")):
        with T.Kernel(2, threads=32) as row:
            encoded = T.view(state, dtype="float8_e4m3fn")
            for column in T.Parallel(4):
                out[row, column] = T.cast(encoded[row, column], "float32")

    mapped = map_state(kernel, (0,), rows=2, capacity=4)
    reads = []
    ir.stmt_functor.post_order_visit(
        mapped.body,
        lambda node: (
            reads.append(node)
            if isinstance(node, ir.BufferLoad)
            and node.buffer.data == kernel.buffer_map[kernel.params[0]].data
            else None
        ),
    )
    assert reads and all(str(read.dtype) == "float8_e4m3fn" for read in reads)
    assert all(tuple(int(n) for n in read.buffer.shape) == (4, 4) for read in reads)


def test_cuda_reinterpreted_state_reads_reordered_slots(tmp_path):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in FP8 view mapping qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.quantization._shared.e4m3 import decode_storage

    entry = tmp_path / "view.py"
    entry.write_text("""
import tilelang.language as T
from akas.compilation.transforms.state import map_state
def tensor_export():
    @T.prim_func
    def kernel(state:T.Tensor((2,4),"uint8"),out:T.Tensor((2,4),"float32")):
        with T.Kernel(2,threads=32) as row:
            encoded=T.view(state,dtype="float8_e4m3fn")
            for column in T.Parallel(4):
                out[row,column]=T.cast(encoded[row,column],"float32")
    return {"kernel":map_state(kernel,(0,),rows=2,capacity=4)}
""")
    artifact = entry.with_suffix(".tbin")
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    values = np.array(
        [[1, 2, 3, 4], [48, 56, 64, 72], [5, 6, 7, 8], [176, 184, 192, 200]],
        dtype="uint8",
    )
    arrays = dict(
        state=values,
        out=np.zeros((2, 4), dtype="float32"),
        mapping=np.array([3, 1], dtype="int32"),
    )
    with tensor.Device(provider="cuda") as device:
        buffers = {name: device.from_numpy(a) for name, a in arrays.items()}
        with device.load(artifact) as kernel:
            kernel.launch(*[buffers[a["name"]] for a in kernel.manifest["arguments"]])
            np.testing.assert_array_equal(
                buffers["out"].to_numpy(), decode_storage(values[[3, 1]])
            )
        for buffer in buffers.values():
            buffer.release()
