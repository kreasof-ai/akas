"""Native dense composition binds through the common artifact/workspace executor."""

import os
from types import SimpleNamespace
import numpy as np
import pytest


def checkpoint():
    from akas.models.qwen35.config import DenseConfig
    from akas.models.qwen35.weights import DenseCheckpoint

    config = DenseConfig(1024,2048,256,("linear_attention","full_attention"),8,2,256,16,32,128,128,1e-6,1e6,64,4096)
    value = object.__new__(DenseCheckpoint)
    value.config = config
    value.tensors = {name:SimpleNamespace(shape=shape,dtype="F32" if name.endswith(("linear_attn.A_log","linear_attn.norm.weight")) else "BF16") for name,shape in config.expected_tensors().items()}
    return value


def test_dense_composition_declares_weights_state_and_commit_without_device():
    from akas.models.qwen35.model import compose_dense

    graph,commit = compose_dense(checkpoint(),slots=2,chunk=4,pool=4,context=64,verify=True)
    assert graph.phase == "prefill"
    assert len(commit.calls) == 2
    assert all(call.parameters.get("commit") is False for call in graph.calls if call.kind in ("gdn_conv","gdn_scan"))
    assert all(graph.buffers[name].role == "weight" for name in checkpoint().storage_specs())
    assert graph.buffers["state.0.0"].shape[0] == 4
    assert graph.buffers["mapping"].shape == (2,)


@pytest.mark.skipif(os.environ.get("TENSOR_QWEN_CUDA") != "1",reason="native graph binding qualification")
def test_dense_graph_executor_matches_native_plan_and_state(tmp_path):
    from contextlib import ExitStack
    from pathlib import Path
    import json
    import tensor
    from akas.graph import Graph
    from akas.models.qwen35.model import compose_dense
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward
    from akas.execution.batch import Batch
    from akas.memory.transfer import write

    loaded = checkpoint()
    current,_ = compose_dense(loaded,slots=2,chunk=4,pool=4,context=64)
    frozen = json.loads((Path(__file__).parent / "fixtures/dense-program-baseline.json").read_text())
    original = Graph.from_metadata(frozen["forward"])
    for name, graph in (("original", original), ("current", current)):
        produce(graph, tmp_path / name, search=False)
    rng = np.random.default_rng(28)
    with tensor.Device() as device, ExitStack() as stack:
        weights = {}
        for name,spec in current.buffers.items():
            if spec.role == "weight":
                value = device.from_numpy(rng.normal(0,0.01,spec.shape).astype("float32"),dtype=spec.dtype)
                weights[name] = value
                stack.callback(value.release)
        runtimes = {}
        for name, graph in (("original", original), ("current", current)):
            supplied = dict(weights)
            for key,spec in graph.buffers.items():
                if spec.role in ("input", "output"):
                    host = np.zeros(spec.shape,dtype="float32" if spec.dtype=="bfloat16" else spec.dtype)
                    value = device.from_numpy(host,dtype=spec.dtype)
                    supplied[key] = value
                    stack.callback(value.release)
            runtimes[name] = stack.enter_context(Forward(tmp_path / name,device,supplied,graphs=True))
        baseline = runtimes["original"]
        for name,values in {"mapping":[3,1],"position":[0,0],"lengths":[4,2],
                            "head_position":[0,0],"head_active":[1,1],
                            "tokens":[1,2,3,4,5,6,0,0]}.items():
            write(device,baseline.buffers[name],np.array(values,dtype="int32"))
        baseline.run()
        runtime = runtimes["current"]
        frame = Batch(runtime,rows=2,chunk=4,inputs=current.request["inputs"],prediction=current.request["prediction"])
        frame.run([3,1],[[1,2,3,4],[5,6]],np.zeros(4,dtype="int32"))
        compared = [name for name,spec in current.buffers.items() if spec.role=="state"] + ["predicted", "normal"]
        for name in compared:
            actual,expected = runtime.buffers[name].to_numpy(),baseline.buffers[name].to_numpy()
            if name == "normal":
                # Padding rows are unobservable scratch: attention leaves
                # their output unwritten. Every live row remains bit exact.
                actual,expected = actual[:6],expected[:6]
            np.testing.assert_array_equal(actual,expected)
        from tensor.compiler.tuning import measure
        timings = {name: measure(device, forward.run, warmup=10, samples=7, repeats=25)
                   for name, forward in runtimes.items()}
        report = Path("build/dense-graph-parity")
        report.mkdir(parents=True, exist_ok=True)
        (report / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")


def test_request_inputs_are_model_independent_and_pad_unused_rows():
    from akas.execution.batch import Batch
    from unittest.mock import patch
    from types import SimpleNamespace

    written = {}
    names = {field:field for field in ("tokens","mapping","positions","lengths","head_positions","head_active")}
    forward = SimpleNamespace(device=object(),buffers={name:SimpleNamespace(name=name,shape=(12,) if name=="tokens" else (4,)) for name in names},run=lambda:None)
    batch = Batch(forward,rows=4,chunk=3,inputs=names,prediction="predicted")
    with patch("akas.execution.batch.write_many",lambda device,updates:written.update({buffer.name:values.copy() for buffer,values in updates})):
        batch.prepare([3,1],[[7,8,9],[4]],np.array([0,5,0,9],dtype="int32"))
    np.testing.assert_array_equal(written["mapping"],[3,1,-1,-1])
    np.testing.assert_array_equal(written["lengths"],[3,1,0,0])
    np.testing.assert_array_equal(written["positions"],[9,5,0,0])
    np.testing.assert_array_equal(written["tokens"],[7,8,9,4,0,0,0,0,0,0,0,0])
    np.testing.assert_array_equal(written["head_positions"],[11,5,0,0])


@pytest.mark.skipif(os.environ.get("TENSOR_QWEN_CUDA") != "1",reason="common executor qualification")
def test_dense_program_runs_public_executor_without_native_binding(tmp_path):
    import tensor
    from akas.compilation.compile import compile_program
    from akas.models.qwen35.model import compose_dense
    from akas.execution.executor import Executor

    loaded = checkpoint()
    def definition(checkpoint, *, rows,chunk,max_requests,max_context,provider,phase=None):
        return compose_dense(checkpoint,slots=rows,chunk=chunk,pool=max_requests,context=max_context,verify=phase=="verify")
    bindings = {"tokens":"tokens","mapping":"mapping","positions":"position",
                "lengths":"lengths","head_positions":"head_position","head_active":"head_active"}
    compile_program(loaded,tmp_path,definition=definition,geometries=[(2,1),(2,4),(2,4,"verify")],
                    max_requests=4,max_context=64,inputs=bindings,prediction="predicted",
                    reset_buffers=("state.0.0","state.0.1"),search=False)
    rng = np.random.default_rng(17)
    def upload(device,**options):
        weights = {}
        for name,(shape,dtype) in loaded.storage_specs().items():
            host = rng.normal(0,0.01,shape).astype("float32")
            weights[name] = device.from_numpy(host,dtype="bfloat16") if dtype=="bfloat16" else device.from_numpy(host)
        return weights
    loaded.upload = upload
    with tensor.Device() as device:
        with Executor.from_program(loaded,tmp_path,device) as executor:
            a,b = executor.admit_at(3),executor.admit_at(1)
            result = executor.run_batch([a,b],[[1,2,3,4],[5,6]],chunk=4)
            assert len(result) == 2
            np.testing.assert_array_equal(executor.positions,[0,2,0,4])
            result = executor.decode([b,a],result[::-1])
            assert len(result) == 2
            np.testing.assert_array_equal(executor.positions,[0,3,0,5])
            verified = executor.verify([a,b],[[7,8,9],[10,11]],chunk=4)
            assert list(map(len,verified)) == [3,2]
            np.testing.assert_array_equal(executor.positions,[0,3,0,5])
            executor.commit([a,b],[2,0])
            np.testing.assert_array_equal(executor.positions,[0,3,0,7])
            executor.verify([a,b],[[12],[13]],chunk=4)
            executor.discard([a,b])
            np.testing.assert_array_equal(executor.positions,[0,3,0,7])
            # Fixed-group replay must produce the same final token and state
            # as scheduled one-step execution, without intermediate readbacks.
            executor.release(a)
            executor.release(b)
            a, b = executor.admit_at(3), executor.admit_at(1)
            initial = executor.run_batch([a,b], [[1,2,3,4],[5,6]], chunk=4)
            expected_tokens = initial
            for _ in range(3):
                expected_tokens = executor.decode([a,b], expected_tokens)
            expected_state = {name: value.to_numpy().copy() for name,value in executor.batches.states.items()}
            executor.release(a)
            executor.release(b)
            a, b = executor.admit_at(3), executor.admit_at(1)
            initial = executor.run_batch([a,b], [[1,2,3,4],[5,6]], chunk=4)
            assert executor.decode_steps([a,b], initial, 3, read=False) is None
            actual = executor.batch(2).buffers[executor.batch(2).prediction].to_numpy()[:2].tolist()
            assert actual == expected_tokens
            for name, value in expected_state.items():
                # Unwritten KV beyond the live prefix belongs to prior rounds.
                if loaded.config.layers[int(name.split(".")[1])] == "linear_attention":
                    np.testing.assert_array_equal(executor.batches.states[name].to_numpy(), value)
            np.testing.assert_array_equal(executor.positions, [0,5,0,7])
            executor.release(a)
            assert executor.admit_at(3) == a
            np.testing.assert_array_equal(executor.positions,[0,5,0,0])
            executor.release(a)
            executor.release(b)
        assert not device._buffers


def moe_checkpoint():
    from akas.models.qwen35.config import Qwen35Config
    from akas.models.qwen35.weights import Qwen35Checkpoint

    config = Qwen35Config(("linear_attention", "full_attention"), 64, 1e-6, 1e6, vocab=1024)
    loaded = object.__new__(Qwen35Checkpoint)
    loaded.config = config
    shapes = config.expected_tensors()
    loaded.tensors = {}
    for name, shape in shapes.items():
        dtype = "F8_E4M3" if name + "_scale_inv" in shapes else "BF16"
        if name.endswith(("linear_attn.A_log", "linear_attn.norm.weight")):
            dtype = "F32"
        nbytes = int(np.prod(shape)) * (1 if dtype == "F8_E4M3" else 4 if dtype == "F32" else 2)
        loaded.tensors[name] = SimpleNamespace(shape=shape, dtype=dtype, nbytes=nbytes)
    return loaded


@pytest.mark.skipif(os.environ.get("TENSOR_QWEN_CUDA") != "1", reason="native MoE program qualification")
def test_moe_program_preserves_row_local_forward_with_sparse_request_pool(tmp_path):
    """Compare unchanged row-local kernels with the common mapped executor."""
    from dataclasses import replace
    from contextlib import ExitStack
    import tensor
    from akas.graph import Graph
    from akas.compilation.compile import compile_program, produce
    from akas.models.qwen35.model import compose_moe_program
    from akas.execution.executor import Executor
    from akas.execution.forward import Forward
    from akas.execution.batch import Batch

    loaded = moe_checkpoint()
    def upload(device, **options):
        rng = np.random.default_rng(41)
        result = {}
        try:
            for name, (shape, dtype) in loaded.storage_specs(pack_experts=True).items():
                if dtype == "uint8":
                    # Finite signed E4M3 payloads without a full-size floating
                    # host temporary for the resident expert matrices.
                    host = rng.integers(16, 40, shape, dtype="uint8")
                    host |= rng.integers(0, 2, shape, dtype="uint8") * 128
                elif name.endswith("scale_inv"):
                    host = np.full(shape, 0.03125, dtype="float32")
                elif name.endswith("A_log"):
                    host = np.full(shape, -3, dtype="float32")
                elif name.endswith("linear_attn.norm.weight"):
                    host = np.ones(shape, dtype="float32")
                else:
                    host = rng.normal(0, 0.02, shape).astype("float32")
                result[name] = device.from_numpy(host, dtype=dtype)
            return result
        except BaseException:
            for value in result.values():
                value.release()
            raise
    loaded.upload = upload
    baseline = {}
    for chunk in (1, 4):
        graph = compose_moe_program(loaded, rows=2, chunk=chunk, max_requests=4, max_context=32)
        graph = Graph.from_metadata(graph.metadata())
        for name, spec in tuple(graph.buffers.items()):
            if spec.role == "state":
                graph.buffers[name] = replace(spec, shape=(2, *spec.shape[1:]))
        del graph.buffers["mapping"]
        graph.request["inputs"].pop("mapping")
        path = tmp_path / f"local-{chunk}"
        produce(graph, path, search=False)
        baseline[chunk] = (path, graph)
    destination = tmp_path / "program"
    compile_program(loaded, destination, definition=compose_moe_program,
                    geometries=[(2, 1), (2, 4)], max_requests=4, max_context=32,
                    upload_options={"pack_experts": True}, search=False)
    with tensor.Device() as device, ExitStack() as stack:
        executor = stack.enter_context(Executor.from_program(loaded, destination, device))
        slots = [executor.admit_at(3), executor.admit_at(1)]
        references, shared = {}, {}
        for chunk in (4, 1):
            path, graph = baseline[chunk]
            supplied = {name: value for name, value in executor.weights.items() if name in graph.buffers}
            supplied.update(shared)
            for name, spec in graph.buffers.items():
                if spec.role in ("input", "output"):
                    host = np.zeros(spec.shape, dtype="float32" if spec.dtype == "bfloat16" else spec.dtype)
                    value = device.from_numpy(host, dtype=spec.dtype)
                    stack.callback(value.release)
                    supplied[name] = value
            forward = stack.enter_context(Forward(path, device, supplied, graphs=True))
            shared.update((name, forward.buffers[name]) for name, spec in graph.buffers.items() if spec.role == "state")
            references[chunk] = Batch(forward, rows=2, chunk=chunk,
                                      inputs=graph.request["inputs"], prediction=graph.request["prediction"])
        for sequences, chunk in (([[1,2,3,4],[5,6]],4), ([[7],[8]],1), ([[9]],1)):
            reference = references[chunk]
            expected = reference.run(list(range(len(sequences))), sequences, np.zeros(2,dtype="int32"))
            actual = executor.run_batch(slots[:len(sequences)], sequences, chunk=chunk)
            assert actual == expected
            for name, state in shared.items():
                result = executor.batches.states[name].to_numpy()
                np.testing.assert_array_equal(result[slots], state.to_numpy())
                np.testing.assert_array_equal(result[[0,2]], 0)
            np.testing.assert_array_equal(executor.batches.states["positions"].to_numpy(), executor.positions)


def test_moe_decode_declares_native_storage_and_common_slot_mapping():
    from akas.models.qwen35.model import compose_moe_decode
    from akas.compilation.compile import plan
    loaded = moe_checkpoint()
    from akas.models.qwen35.model import compose_moe_program
    prefill = compose_moe_program(loaded, rows=2, chunk=4, max_requests=4, max_context=32)
    prefill_bundle = plan(prefill)
    assert prefill.request["inputs"]["tokens"] == "chunk.tokens"
    assert any(kernel.kind == "quantize" for kernel in prefill_bundle.kernels)
    assert any(kernel.kind == "fp8_experts" and kernel.source.module == "akas.ops.moe.cuda.prefill" for kernel in prefill_bundle.kernels)
    graph = compose_moe_decode(loaded, rows=2, max_requests=4, max_context=32)
    bundle = plan(graph)
    assert graph.request["prediction"] == "tokens"
    assert graph.buffers["positions"].shape == (4,)
    assert graph.buffers["state.0.0"].shape[0] == 4
    assert graph.buffers["state.1.0"].shape[0] == 4
    assert any(kernel.mapping is not None for kernel in bundle.kernels if kernel.kind == "gdn_recurrent")
    assert any(kernel.mapping is not None for kernel in bundle.kernels if kernel.kind == "argmax")
    assert not any(".experts.0." in name for name in graph.buffers)


@pytest.mark.skipif(os.environ.get("TENSOR_QWEN_CUDA") != "1", reason="automatic native MoE qualification")
@pytest.mark.parametrize("chunk", [1, 4])
def test_moe_phases_inherit_kernel_search_without_model_tuning_hooks(tmp_path, chunk):
    from akas.models.qwen35.model import compose_moe_program
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings

    graph = compose_moe_program(moe_checkpoint(), rows=2, chunk=chunk,
                                max_requests=4, max_context=32)
    manifest = produce(graph, tmp_path,
                       search_settings=SearchSettings(budget=2, samples=1, repeats=1))
    assert manifest["tuning"]["operations"]
    assert all(row["candidates"][0]["status"] == "passed" for row in manifest["tuning"]["operations"])


def test_proposal_graph_declares_shared_embedding_and_separate_head_state():
    from akas.models.qwen35.model import compose_dense

    graph, _ = compose_dense(checkpoint(), slots=2, chunk=4, pool=4, context=64, draft=True)
    assert "model.language_model.embed_tokens.weight" in graph.buffers
    assert "mtp.fc.weight" in graph.buffers
    assert "proposal.saved_hidden" in graph.buffers
    assert "state.0.0" not in graph.buffers
    assert graph.request["features"]["hidden_input"] == "proposal.hidden"
    assert graph.request["features"]["mode"] == "proposal.mode"
    assert graph.request["inputs"]["tokens"] == "proposal.tokens"
    assert checkpoint().storage_specs(branch="all").keys() == (checkpoint().storage_specs().keys() | checkpoint().storage_specs(branch="mtp").keys())


@pytest.mark.skipif(os.environ.get("TENSOR_QWEN_CUDA") != "1", reason="shared executor MTP qualification")
@pytest.mark.parametrize("capture,deferred", [(False,False),(True,False),(True,True)])
def test_mtp_head_uses_common_verification_and_repair_without_native_runtime(tmp_path, capture, deferred):
    import tensor
    from akas.compilation.compile import compile_program
    from akas.models.qwen35.model import compose_dense
    from akas.execution.executor import Executor
    from akas.models.heads.mtp import MTPHead
    from akas.generation.speculative.decoding import SpeculativeDecoder

    loaded = checkpoint()
    def definition(checkpoint, *, rows, chunk, max_requests, max_context, provider, phase=None):
        return compose_dense(checkpoint, slots=rows, chunk=chunk, pool=max_requests,
                             context=max_context, verify=phase=="verify", draft=phase=="proposal")
    compile_program(loaded, tmp_path, definition=definition,
                    geometries=[(2,1),(2,4),(2,4,"verify"),(2,1,"proposal"),(2,4,"proposal")],
                    max_requests=4, max_context=64, upload_options={"branch":"all"}, round_windows=(4,), deferred=deferred, search=False)
    def upload(device, **options):
        rng = np.random.default_rng(53)
        result = {}
        for name, (shape, dtype) in loaded.storage_specs(branch="all").items():
            values = rng.normal(0, 0.01, shape).astype("float32")
            result[name] = device.from_numpy(values, dtype="bfloat16") if dtype=="bfloat16" else device.from_numpy(values)
        return result
    loaded.upload = upload
    with tensor.Device() as device:
        with Executor.from_program(loaded, tmp_path, device) as target, Executor.from_program(loaded, tmp_path, device) as baseline:
            head = MTPHead(target)
            decoder = SpeculativeDecoder(target, head, window=4, capture=capture, deferred=deferred)
            requests = []
            for prompt in ([1,2,3,4],[5,6]):
                slot = decoder.admit()
                baseline.admit_at(slot)
                requests.append(SimpleNamespace(slot=slot, prompt=prompt, offset=0, output=[], limit=12))
            first = decoder.prefill_batch(requests, [request.prompt for request in requests], 4)
            assert first == baseline.prefill([request.slot for request in requests], [request.prompt for request in requests], chunk=4)
            for request, token in zip(requests, first):
                request.output.append(token)
            for _ in range(2):
                first_expected = baseline.decode([request.slot for request in requests], [request.output[-1] for request in requests])
                head.cached.update((request.slot, token) for request,token in zip(requests,first_expected))
                emitted = decoder.decode_many(requests)
                assert all(len(tokens) >= 2 for tokens in emitted)
                expected = [[token] for token in first_expected]
                previous = list(first_expected)
                for index in range(1,max(map(len, emitted))):
                    active = [row for row, tokens in enumerate(emitted) if index < len(tokens)]
                    predicted = baseline.decode([requests[row].slot for row in active], [previous[row] for row in active])
                    for row, token in zip(active, predicted):
                        expected[row].append(token)
                        previous[row] = token
                assert emitted == expected
                for request, tokens in zip(requests, emitted):
                    request.output.extend(tokens)
                np.testing.assert_array_equal(target.positions, baseline.positions)
                np.testing.assert_array_equal(head.positions, target.positions)
                assert not target.pending
            target._materialize([request.slot for request in requests])
            np.testing.assert_allclose(target.batches.states["state.0.0"].to_numpy(), baseline.batches.states["state.0.0"].to_numpy(), rtol=1e-5, atol=1e-6)
            for request in requests:
                decoder.release(request.slot)
                baseline.release(request.slot)
            decoder.close()
        assert not device._buffers


def test_direct_moe_composition_preserves_frozen_operation_and_storage_contracts():
    import json
    from pathlib import Path
    from akas.models.qwen35.model import compose_moe_program

    fixture = json.loads((Path(__file__).parent / "fixtures/moe-program-baseline.json").read_text())
    loaded = moe_checkpoint()
    for chunk in (1, 4):
        expected = fixture["geometries"][str(chunk)]
        actual = json.loads(json.dumps(compose_moe_program(
            loaded, rows=2, chunk=chunk, max_requests=4, max_context=32).metadata()))
        # The final diagnostic label follows actual depth instead of a fixed
        # forty-layer constant. Kernel arithmetic and call order are unchanged.
        for call in expected["calls"]:
            if call["label"] == [40, "final"]:
                call["label"] = [len(loaded.config.layers), "final"]
        assert actual["calls"] == expected["calls"]
        hidden = actual["request"]["features"]["hidden"]
        for buffer in expected["buffers"]:
            if buffer["name"] == hidden:
                # Exported hidden rows must survive forward completion for a
                # proposal head; they can no longer be recycled as scratch.
                buffer["role"] = "output"
        assert {b["name"]: b for b in actual["buffers"]} == {b["name"]: b for b in expected["buffers"]}
        assert actual["request"]["inputs"] == expected["request"]["inputs"]
        assert actual["request"]["prediction"] == expected["request"]["prediction"]
        assert set(actual["request"]["reset"]) == set(expected["request"]["reset"])
