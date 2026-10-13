"""Weight formats do not choose request-state addressing."""

import os
import numpy as np
import pytest
from pathlib import Path
from akas.compilation.emit import export_kernel
from akas.compilation.emit import build_artifact


def test_state_mapping_rejects_a_copy_spanning_multiple_requests():
    import tilelang.language as T
    from akas.ops.state.mapping import map_state

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
            "akas.ops.state.mapping",
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
    from tensor.compiler.frontend.source import export_source
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    rows, pool = 2, 4
    if kind == "gdn_recurrent":
        module = "akas.ops.gdn.cuda.decode"
        p = dict(slots=rows,heads=2,key=32,value=32,value_tile=16)
        states = (6,)
    else:
        module = "akas.ops.attention.cuda.prefill"
        p = dict(slots=rows,chunk=2,capacity=64)
        states = (1,2)
    paths = []
    for index in range(2):
        path = tmp_path/f'kernel-{index}.py'
        text = (export_source(module,"make_kernel",kind,p,dependencies=()) if index==0 else
                export_kernel("akas.ops.state.mapping","make_kernel",module,"make_kernel",(kind,p),states,rows,pool))
        path.write_text(text)
        image = path.with_suffix('.tbin')
        build_artifact(path,image,target="sm_89")
        paths.append(image)
    rng = np.random.default_rng(34)
    mapping = np.array([3,1],dtype="int32")
    with tensor.Device() as device:
        baseline, mapped = [device.load(path) for path in paths]
        initial, expected_pool, inputs, full_inputs = {}, {}, [], []
        for index, spec in enumerate(baseline.manifest["arguments"]):
            shape, dtype = tuple(spec["shape"]), spec["dtype"]
            full_shape = (pool,*shape[1:]) if index in states else shape
            if dtype == "int32":
                host = np.full(full_shape, 2 if spec["name"]=="lengths" else 3 if spec["name"]=="positions" else 1,dtype="int32")
            else:
                host = rng.normal(0,0.05,full_shape).astype("float32")
                if dtype == "bfloat16": host = decode_bfloat16(encode_bfloat16(host))
            if index in states:
                expected_pool[index] = host.copy()
                small = host[mapping].copy()
            else: small = host.copy()
            inputs.append(device.from_numpy(small,dtype="bfloat16") if dtype=="bfloat16" else device.from_numpy(small))
            full_inputs.append(device.from_numpy(host,dtype="bfloat16") if dtype=="bfloat16" else device.from_numpy(host))
        request_map = device.from_numpy(mapping)
        baseline.launch(*inputs)
        mapped.launch(*full_inputs,request_map)
        np.testing.assert_array_equal(inputs[-1].to_numpy(),full_inputs[-1].to_numpy())
        for index in states:
            expected_pool[index][mapping] = inputs[index].to_numpy()
            np.testing.assert_array_equal(full_inputs[index].to_numpy(),expected_pool[index])
