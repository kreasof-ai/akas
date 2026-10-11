"""Bit-preserving checkpoint transfers with cleanup on partial failure."""

import ctypes as ct
import mmap
from types import MappingProxyType

import numpy as np

from akas.checkpoints.safetensors import upload_bits

DTYPES = {"BF16": "bfloat16", "F32": "float32", "F8_E4M3": "uint8"}


def upload_packed(device, checkpoint):
    weights = {}
    try:
        for name, info in checkpoint.tensors.items():
            raw = checkpoint.packed(name)
            if info.type in (0, 1):
                raw = raw.view(np.float32 if info.type == 0 else np.float16)
            elif device.info["provider"] == "webgpu":
                raw = raw.view(np.uint32)
            weights[name] = device.from_numpy(np.ascontiguousarray(raw))
    except BaseException:
        for buffer in weights.values():
            buffer.release()
        raise
    return weights


def upload_shards(device, tensors, wanted, layouts):
    """Map one shard at a time and copy bytes into declared grouped layouts."""
    buffers = {}
    try:
        for name in wanted:
            group, _, shape = layouts[name]
            if group not in buffers:
                buffers[group] = device.empty(shape, DTYPES[tensors[name].dtype])
        for shard in sorted({tensors[name].shard for name in wanted}):
            with (
                shard.open("rb") as stream,
                mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
            ):
                for name in wanted:
                    info = tensors[name]
                    if info.shard != shard:
                        continue
                    group, offset, _ = layouts[name]
                    view = np.frombuffer(
                        mapped, dtype="uint8", count=info.nbytes, offset=info.offset
                    )
                    try:
                        device._check()
                        device.driver.call(
                            "cuMemcpyHtoD_v2",
                            buffers[group].pointer + offset,
                            ct.c_void_p(view.ctypes.data),
                            info.nbytes,
                        )
                        device.driver.call("cuStreamSynchronize", None)
                    finally:
                        del view
        return MappingProxyType(buffers)
    except BaseException:
        for buffer in buffers.values():
            buffer.release()
        raise


def upload_concatenated(device, tensors, read, groups):
    """Upload native tensors and concatenate declared adjacent projections."""
    weights = {}
    try:
        for name, info in tensors.items():
            weights[name] = upload_bits(device, read(name), DTYPES[info.dtype])
        for destination, names, dtype in groups:
            weights[destination] = upload_bits(
                device, np.concatenate([read(name) for name in names]), dtype
            )
            for name in names:
                weights.pop(name).release()
        return weights
    except BaseException:
        for buffer in weights.values():
            buffer.release()
        raise
