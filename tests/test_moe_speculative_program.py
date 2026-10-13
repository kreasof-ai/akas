"""MoE MTP uses the same proposal and accepted-prefix engine as dense models."""

import os
from types import SimpleNamespace
import numpy as np
import pytest


def checkpoint():
    from test_qwen_graph import moe_checkpoint
    from akas.models.qwen35.weights import Qwen35Checkpoint

    loaded = moe_checkpoint()
    mtp = object.__new__(Qwen35Checkpoint)
    mtp.config, mtp.branch = loaded.config, "mtp"
    shapes = loaded.config.expected_mtp_tensors()
    mtp.tensors = {}
    for name, shape in shapes.items():
        dtype = "F8_E4M3" if name + "_scale_inv" in shapes else "BF16"
        mtp.tensors[name] = SimpleNamespace(
            shape=shape,
            dtype=dtype,
            nbytes=int(np.prod(shape)) * (1 if dtype == "F8_E4M3" else 2),
        )
    loaded.mtp_checkpoint = mtp

    def upload(device, **options):
        rng = np.random.default_rng(41)
        result = {}
        try:
            for name, (shape, dtype) in loaded.storage_specs(
                pack_experts=True, branch="all"
            ).items():
                if dtype == "uint8":
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
    return loaded


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    from akas.compilation.compile import compile_program
    from akas.models.qwen35.model import compose_moe_program

    loaded = checkpoint()
    reused = os.environ.get("AKAS_TEST_MOE_PROGRAM")
    if reused:
        from dataclasses import asdict
        from pathlib import Path
        import json
        from akas.artifacts.read import read_program

        path = Path(reused)
        manifest = read_program(path)
        assert manifest["config"] == json.loads(json.dumps(asdict(loaded.config)))
        return loaded, path
    path = tmp_path_factory.mktemp("moe-mtp")
    compile_program(
        loaded,
        path,
        definition=compose_moe_program,
        geometries=[
            (2, 1),
            (2, 4),
            (2, 4, "verify"),
            (2, 1, "proposal"),
            (2, 4, "proposal"),
        ],
        max_requests=4,
        max_context=32,
        round_windows=(4,),
        search=False,
    )
    return loaded, path


@pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1", reason="native MoE MTP qualification"
)
@pytest.mark.parametrize("capture", [False, True])
def test_moe_proposal_and_repair_match_common_greedy_executor(program, capture):
    import tensor
    from akas.execution.executor import Executor
    from akas.models.heads.mtp import MTPHead
    from akas.generation.speculative.decoding import SpeculativeDecoder

    loaded, path = program
    with tensor.Device() as device:
        with (
            Executor.from_program(loaded, path, device) as target,
            Executor.from_program(loaded, path, device) as baseline,
        ):
            head = MTPHead(target)
            decoder = SpeculativeDecoder(target, head, window=4, capture=capture)
            requests = []
            for slot, prompt in ((3, [1, 2, 3, 4]), (1, [5, 6])):
                target.admit_at(slot)
                head.admit(slot)
                baseline.admit_at(slot)
                requests.append(
                    SimpleNamespace(
                        slot=slot, prompt=prompt, offset=0, output=[], limit=12
                    )
                )
            first = decoder.prefill_batch(
                requests, [request.prompt for request in requests], 4
            )
            assert first == baseline.prefill(
                [request.slot for request in requests],
                [request.prompt for request in requests],
                chunk=4,
            )
            for request, token in zip(requests, first):
                request.output.append(token)
            expected_first = baseline.decode(
                [request.slot for request in requests],
                [request.output[-1] for request in requests],
            )
            head.cached.update(
                (request.slot, token)
                for request, token in zip(requests, expected_first)
            )
            emitted = decoder.decode_many(requests)
            expected = [[token] for token in expected_first]
            previous = list(expected_first)
            for depth in range(1, max(map(len, emitted))):
                active = [
                    row for row, tokens in enumerate(emitted) if depth < len(tokens)
                ]
                predicted = baseline.decode(
                    [requests[row].slot for row in active],
                    [previous[row] for row in active],
                )
                for row, token in zip(active, predicted):
                    expected[row].append(token)
                    previous[row] = token
            assert emitted == expected
            np.testing.assert_array_equal(target.positions, baseline.positions)
            np.testing.assert_array_equal(head.positions, target.positions)
            decoder.close()
