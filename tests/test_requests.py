"""Common request isolation, ownership, loading failures and real CUDA serving."""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from akas import Engine, EngineConfig, LFM2
from akas.benchmarking.session import Session


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_request_limit_rejected_before_loading(limit):
    with pytest.raises(ValueError, match="capacity"):
        LFM2("missing-model", "missing-bundle", SimpleNamespace(), capacity=limit)


GPU = pytest.mark.skipif(os.environ.get("TENSOR_LFM2_CUDA") != "1", reason="enable native CUDA request qualification")


@pytest.fixture(scope="module")
def fixture_bundle(tmp_path_factory):
    from support.diagnostic import fixture
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    directory = Path(os.environ["TENSOR_LFM2_REQUEST_CACHE"]) if os.environ.get("TENSOR_LFM2_REQUEST_CACHE") else tmp_path_factory.mktemp("common-requests")
    directory.mkdir(parents=True, exist_ok=True)
    model = fixture(directory / "model.gguf")
    bundle = directory / "bundle"
    compile_checkpoint(LFM2, model, bundle, checkpoint_options={"tokenizer": False},
                       geometries=[(2, 1), (2, 32), (2, 128)], max_requests=8, max_context=384,
                       search_settings=SearchSettings(budget=1, samples=1, repeats=1))
    return model, bundle


def feature(model, chunk):
    frame = model.batches.select(1, chunk)
    return frame.feature("logits").to_numpy().reshape(frame.rows, -1)[0].copy()


@GPU
@pytest.mark.parametrize("graphs", [False, True])
def test_eight_interleaved_requests_and_slot_lifecycle(fixture_bundle, graphs):
    import tensor
    path, bundle = fixture_bundle
    prefixes = [[256] + [(j + 13*i) % 256 for j in range(n-1)] for i,n in enumerate((1,2,31,32,127,128,129,257))]
    with tensor.Device() as device, LFM2(path, bundle, device, graphs=graphs) as model:
        slots = [model.admit() for _ in range(8)]
        weights = dict(model.weights)
        with pytest.raises(ValueError, match="exhausted"):
            model.admit()
        with LFM2(path, bundle, device, graphs=graphs) as independent:
            expected = []
            for prefix in prefixes:
                slot = independent.admit()
                independent.prefill([slot], [prefix], chunk=128)
                row = [feature(independent, independent.last_prefill_chunk)]
                independent.decode([slot], [65]); row.append(feature(independent, 1))
                independent.prefill([slot], [[66,67]], chunk=32); row.append(feature(independent, independent.last_prefill_chunk))
                expected.append(row)
                independent.release(slot)
            for index, slot in enumerate(slots):
                model.prefill([slot], [prefixes[index]], chunk=128)
                np.testing.assert_array_equal(feature(model, model.last_prefill_chunk), expected[index][0])
            for index in reversed(range(8)):
                model.decode([slots[index]], [65])
                np.testing.assert_array_equal(feature(model,1), expected[index][1])
            for index in range(8):
                model.prefill([slots[index]], [[66,67]], chunk=32)
                np.testing.assert_array_equal(feature(model, model.last_prefill_chunk), expected[index][2])
        model.release(slots[3])
        assert model.admit() == slots[3] and model.positions[slots[3]] == 0
        assert all(model.weights[name] is value for name,value in weights.items())
        for slot in slots:
            model.release(slot)
        assert not model.occupied.any()
    assert not device._buffers


@GPU
@pytest.mark.parametrize("operation", ["from_numpy", "load"])
def test_failed_shared_model_loading_releases_allocations(fixture_bundle, monkeypatch, operation):
    import tensor
    path,bundle = fixture_bundle
    with tensor.Device() as device:
        original = getattr(device,operation)
        created=[]
        def fail(*args,**kwargs):
            if len(created)==2:
                raise MemoryError("injected loading failure")
            result=original(*args,**kwargs); created.append(result); return result
        with monkeypatch.context() as patch:
            patch.setattr(device,operation,fail)
            with pytest.raises(MemoryError,match="injected"):
                with LFM2(path,bundle,device,checkpoint_options={"tokenizer":False}) as model:
                    model.admit()
        assert not device._buffers and not device._executables
        assert all(value._released for value in created)


@GPU
def test_failed_phase_capture_keeps_shared_weights_and_admission_consistent(fixture_bundle, monkeypatch):
    import tensor
    import tensor.providers.cuda.graph as module
    path,bundle=fixture_bundle
    with tensor.Device() as device, LFM2(path,bundle,device) as model:
        first=model.admit()
        original=module.CudaGraph
        def fail(*args,**options):
            raise RuntimeError("injected capture failure")
        with monkeypatch.context() as patch:
            patch.setattr(module,"CudaGraph",fail)
            with pytest.raises(RuntimeError,match="injected"):
                model.prefill([first],[[256,65]],chunk=32)
        assert model.occupied[first] and model.positions[first]==0
        assert all(not value._released for value in model.weights.values())
        monkeypatch.setattr(module,"CudaGraph",original)
        assert len(model.prefill([first],[[256,65]],chunk=32))==1
        model.release(first)
    assert not device._buffers


@GPU
def test_reject_wrong_thread_before_state_mutation(fixture_bundle):
    import tensor
    path,bundle=fixture_bundle
    with tensor.Device() as device, LFM2(path,bundle,device) as model:
        slot=model.admit()
        before=model.positions.copy()
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError,match="thread"):
                pool.submit(model.decode,[slot],[65]).result()
        np.testing.assert_array_equal(model.positions,before)
        model.release(slot)


@GPU
def test_checksum_config_and_capacity_rejections_before_upload(fixture_bundle, monkeypatch):
    import tensor
    from akas.models.lfm2.weights import LFM2Checkpoint
    path,bundle=fixture_bundle
    with tensor.Device() as device:
        def forbidden(*args,**options):
            raise AssertionError("rejection must precede upload")
        monkeypatch.setattr(LFM2Checkpoint,"upload",forbidden)
        with pytest.raises(ValueError,match="max_requests"):
            LFM2(path,bundle,device,capacity=9)
        from akas.artifacts.read import read_program
        manifest=read_program(bundle)
        descriptor=manifest["phases"][0]
        phase=bundle/descriptor["path"]
        data=phase.read_bytes()
        try:
            phase.write_bytes(data+b' ')
            with pytest.raises(ValueError,match="checksum"):
                LFM2(path,bundle,device)
        finally:
            phase.write_bytes(data)
        assert not device._buffers


@GPU
def test_gpu_host_sampling_and_independent_numeric_reference(fixture_bundle):
    import tensor
    path,bundle=fixture_bundle
    with tensor.Device() as device, LFM2(path,bundle,device) as model:
        device_result=model.generate("Hello",max_tokens=4,chat=False,sampling="device")
        host_result=model.generate("Hello",max_tokens=4,chat=False,sampling="host")
        assert device_result["generated_tokens"]==host_result["generated_tokens"]
        assert not model.occupied.any()


@GPU
def test_shared_engine_schedules_and_reclaims_request_slots(fixture_bundle):
    import tensor
    path,bundle=fixture_bundle
    with tensor.Device() as device, Engine.from_checkpoint(LFM2,path,bundle,device,config=EngineConfig(prefill_chunk=32)) as engine:
        first=engine.submit([256,65,66],max_tokens=4)
        second=engine.submit([256,67],max_tokens=2)
        while engine.step():
            pass
        assert len(first.output)==4 and len(second.output)==2
        assert not engine.model.occupied.any()
        replacement=engine.submit([256,65,66],max_tokens=4)
        while engine.step():
            pass
        assert replacement.output==first.output
    assert not device._buffers


@GPU
def test_shared_http_serves_lfm2_with_real_cuda_resources(fixture_bundle):
    from aiohttp.test_utils import TestClient,TestServer
    from akas.serving import create_app
    from support.serving import lfm2
    path,bundle=fixture_bundle
    async def check():
        async with TestClient(TestServer(create_app(partial(lfm2,checkpoint=path,bundle=bundle,capacity=8)))) as client:
            info=await (await client.get('/server_info')).json()
            assert info['capacity']==8
            response=await client.post('/generate',json=dict(input_ids=[256,65],sampling_params=dict(max_new_tokens=3,temperature=0,ignore_eos=True)))
            assert response.status==200
            text=await asyncio.wait_for(response.text(),20)
            records=[json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
            assert len(records[-1]['output_ids'])==3
    asyncio.run(check())
