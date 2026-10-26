"""Sharing phase resources preserves outputs, integrity and ownership."""

import hashlib
import os
from types import SimpleNamespace
import numpy as np
import pytest


def test_kernel_cache_keeps_integrity_checks(tmp_path):
    from akas.artifacts.read import load_kernels

    file = tmp_path / "kernel.tbin"
    file.write_bytes(b"compiled")
    row = dict(path=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    loaded = []

    class Device:
        def load(self, path):
            obj = SimpleNamespace(released=False)
            obj.release = lambda: setattr(obj, "released", True)
            loaded.append(obj)
            return obj

    cache = {}
    a = load_kernels(Device(), tmp_path, {"first": row}, cache=cache)
    b = load_kernels(Device(), tmp_path, {"second": row}, cache=cache)
    assert a["first"] is b["second"] and len(loaded) == 1
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_kernels(Device(), tmp_path, {"third": row}, cache=cache)
    assert not loaded[0].released


@pytest.mark.skipif(
    os.getenv("AKAS_SPARK_CUDA") != "1", reason="opt-in CUDA arena ownership"
)
def test_weight_arena_releases_after_last_slice_and_keeps_encoded_bytes():
    import tensor
    from akas.memory.arena import UploadArena

    first = np.arange(17, dtype="float32")
    encoded = np.arange(123, dtype="uint8")
    with tensor.Device() as device:
        arena = UploadArena(device, [first.nbytes, encoded.nbytes, 64])
        a, b = arena.upload(first), arena.upload(encoded)
        z = arena.empty((32,), "float16", zero=True)
        np.testing.assert_array_equal(z.to_numpy(), np.zeros(32, dtype="float16"))
        root = arena.root
        assert a.pointer == root.pointer and b.pointer == root.pointer + 256
        assert a._base is b._base
        np.testing.assert_array_equal(a.to_numpy(), first)
        with pytest.raises(ValueError, match="capacity"):
            arena.upload(np.zeros(1024, dtype="uint8"))
        arena.close()
        a.release()
        assert not root._released
        np.testing.assert_array_equal(b.to_numpy(), encoded)
        b.release()
        assert not root._released
        z.release()
        assert root._released
        arena.close()


@pytest.mark.skipif(os.getenv("AKAS_SPARK_CUDA") != "1", reason="opt-in CUDA ownership")
def test_shared_scratch_preserves_outputs_and_other_phase_lifetime(tmp_path):
    import tensor
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.execution.forward import Forward
    from akas.execution.workspace import ScratchPool, release

    for n in (17, 64):
        g = Graph(contract="reference")
        x = g.buffer("x", (n,), "float32", role="input")
        a, b = [g.buffer(s, (n,), "float32") for s in ("a", "b")]
        y = g.buffer("y", (n,), "float32", role="output")
        for lhs, rhs, out in ((x, x, a), (a, x, b), (b, x, y)):
            g("add", dict(r=1, c=n), lhs, rhs, out)
        produce(
            g,
            tmp_path / str(n),
            provider="cuda",
            target="sm_89",
            search_settings=SearchSettings(budget=1, samples=1, repeats=1),
        )
    with tensor.Device() as device:
        pool, cache, frames, inputs, outputs = (
            ScratchPool(device, [256, 256]),
            {},
            [],
            [],
            [],
        )
        try:
            for n in (17, 64):
                x = device.from_numpy(np.arange(n, dtype="float32"))
                y = device.empty((n,), "float32")
                inputs.append(x)
                outputs.append(y)
                frames.append(
                    Forward(
                        tmp_path / str(n),
                        device,
                        {"x": x, "y": y},
                        graphs=True,
                        scratch_pool=pool,
                        kernel_cache=cache,
                    )
                )
            assert frames[0].buffers["a"].pointer == frames[1].buffers["a"].pointer
            frames[0].run()
            frames[1].run()
            for n, y in zip((17, 64), outputs):
                np.testing.assert_array_equal(
                    y.to_numpy(), 4 * np.arange(n, dtype="float32")
                )
            frames[0].close()
            assert not next(iter(pool.buffers.values()))._released
            frames[1].run()
            np.testing.assert_array_equal(
                outputs[1].to_numpy(), 4 * np.arange(64, dtype="float32")
            )
        finally:
            for frame in reversed(frames):
                frame.close()
            release(kernels=cache.values())
            pool.close()
            release(buffers=(*inputs, *outputs))


@pytest.mark.skipif(
    os.getenv("AKAS_SPARK_CUDA") != "1", reason="opt-in CUDA mapped weights"
)
def test_embedding_reads_original_bf16_bytes_from_host(tmp_path):
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.memory.mapped import from_numpy
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact

    p = dict(r=3, c=64, v=16, type=30, ggml=True, start=0)
    script, artifact = tmp_path / "embedding.py", tmp_path / "embedding.tbin"
    script.write_text(
        source(
            "embedding", p, contract="reference", provider="cuda", target="sm_89"
        ).render()
    )
    build_artifact(script, artifact, provider="cuda", target="sm_89")
    meta, _ = read_artifact(artifact)
    x = np.linspace(-1, 1, 16 * 64, dtype="float32")
    bits = (x.view("uint32") >> 16).astype("uint16")
    expected = (bits.astype("uint32") << 16).view("float32").reshape(16, 64)
    with tensor.Device() as device:
        weights = from_numpy(device, bits.view("uint8"))
        tokens = device.from_numpy(np.array([0, 5, 15], dtype="int32"))
        out = device.empty((3 * 64,), "float32")
        with device.load(artifact) as kernel:
            buffers = dict(tokens=tokens, w=weights, out=out)
            kernel.launch(*(buffers[a["name"]] for a in meta["arguments"]))
            np.testing.assert_array_equal(
                out.to_numpy().reshape(3, 64), expected[[0, 5, 15]]
            )
        np.testing.assert_array_equal(weights.to_numpy(), bits.view("uint8"))
