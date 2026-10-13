"""Device token feedback keeps greedy replay independent of model and format."""

import os

import numpy as np
import pytest


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("advance", [False, True])
def test_feedback_is_qualified_by_default_and_replays_without_host_tokens(tmp_path, provider, advance):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable native feedback qualification")
    import tensor
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward

    graph = Graph(contract="foreign")
    arrays = dict(predicted=np.array([11, 12, 13, 14], dtype="int32"),
                  tokens=np.array([1, 2, 3, 4], dtype="int32"),
                  lengths=np.array([1, 0, 1, 0], dtype="int32"))
    if advance:
        arrays["positions"] = np.array([7, 8, 9, 10], dtype="int32")
    arguments = [graph.buffer(name, values.shape, "int32", role="input") for name, values in arrays.items()]
    graph("token_feedback", dict(rows=4, advance=advance), *arguments)
    manifest = produce(graph, tmp_path, provider=provider)
    measurements = manifest["tuning"]["operations"][0]["candidates"]
    assert measurements
    assert all(row["status"] == "passed" for row in measurements)
    with tensor.Device(provider=provider) as device:
        buffers = {name: device.from_numpy(values) for name, values in arrays.items()}
        try:
            with Forward(tmp_path, device, buffers, graphs=True) as forward:
                for _ in range(3):
                    forward.run()
                np.testing.assert_array_equal(buffers["tokens"].to_numpy(), [11, 2, 13, 4])
                if advance:
                    np.testing.assert_array_equal(buffers["positions"].to_numpy(), [10, 8, 12, 10])
        finally:
            for buffer in buffers.values():
                buffer.release()
