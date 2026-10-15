"""Explicit resident storage conversion preserves values and grouped offsets."""

import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pytest

from akas.checkpoints.upload import convert_storage
from akas.checkpoints.upload import upload_shards


def test_e4m3_resident_conversion_is_exact_for_signed_normal_and_subnormal_values():
    raw = np.array([0, 1, 7, 8, 56, 64, 126, 128, 184, 254], dtype="uint8")
    expected = np.array([0, 1/512, 7/512, 1/64, 1, 2, 448, -0., -1, -448], dtype="float16")
    actual = convert_storage(raw, "F8_E4M3", "float16").view("float16")
    np.testing.assert_array_equal(actual.view("uint16"), expected.view("uint16"))
    with pytest.raises(ValueError, match="finite"):
        convert_storage(np.array([127], dtype="uint8"), "F8_E4M3", "float16")


@pytest.mark.parametrize("bad", [False, True])
def test_grouped_conversion_scales_offsets_and_cleans_up_partial_failure(tmp_path, bad):
    path = tmp_path / "shard"
    path.write_bytes(bytes([56, 184, 64, 127 if bad else 192]))
    tensors = {name: SimpleNamespace(dtype="F8_E4M3", shard=path, nbytes=2, offset=index*2)
               for index, name in enumerate(("one", "two"))}
    layouts = {name: ("experts", index*2, (2, 2)) for index, name in enumerate(tensors)}
    allocated = []
    def empty(shape, dtype):
        storage = ct.create_string_buffer(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        value = SimpleNamespace(pointer=ct.addressof(storage), storage=storage, released=False)
        def release():
            value.released = True
        value.release = release
        allocated.append(value)
        return value
    def call(name, *arguments):
        if name == "cuMemcpyHtoD_v2":
            ct.memmove(*arguments)
    device = SimpleNamespace(empty=empty, _check=lambda: None, driver=SimpleNamespace(call=call))
    if bad:
        with pytest.raises(ValueError, match="finite"):
            upload_shards(device, tensors, list(tensors), layouts, conversions={"experts": "float16"})
        assert all(value.released for value in allocated)
    else:
        weights = upload_shards(device, tensors, list(tensors), layouts, conversions={"experts": "float16"})
        actual = np.frombuffer(weights["experts"].storage.raw[:8], dtype="float16")
        np.testing.assert_array_equal(actual, [1, -1, 2, -2])
