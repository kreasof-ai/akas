"""Bit-preserving checkpoint transfers with cleanup on partial failure."""

import ctypes as ct
import mmap
from types import MappingProxyType
import numpy as np
from akas.checkpoints.safetensors import upload_bits

DTYPES = {"BF16": "bfloat16", "F32": "float32", "F16": "float16", "F8_E4M3": "uint8"}


class UploadBatch:
    """Bound asynchronous WebGPU staging; CUDA transfers already complete inline."""

    def __init__(self, device, limit=64 * 1024**2):
        self.device, self.limit, self.pending = device, limit, 0
        self.plan = None

    def __enter__(self):
        if self.device.info["provider"] == "webgpu":
            self.plan = self.device.prepare_plan(())
        return self

    def add(self, size):
        if self.plan is not None:
            self.pending += size
            if self.pending >= self.limit:
                self.flush()

    def flush(self):
        if self.plan is not None and self.pending:
            # An empty public plan submits queued writes without running a kernel.
            self.plan.launch()
            self.device.synchronize()
            self.pending = 0

    def __exit__(self, *error):
        try:
            self.flush()
        finally:
            if self.plan is not None:
                self.plan.close()


def upload_packed(device, checkpoint):
    weights = {}
    try:
        with UploadBatch(device) as transfers:
            for name, info in checkpoint.tensors.items():
                raw = checkpoint.packed(name)
                if info.type in (0, 1):
                    raw = raw.view(np.float32 if info.type == 0 else np.float16)
                elif device.info["provider"] == "webgpu":
                    from akas.quantization._shared.packing import pack_words

                    raw = pack_words(raw)
                weights[name] = device.from_numpy(np.ascontiguousarray(raw))
                transfers.add(raw.nbytes)
                checkpoint.discard(name)
    except BaseException:
        for buffer in weights.values():
            buffer.release()
        raise
    return weights


def convert_storage(raw, source_dtype, target_dtype):
    """Convert explicitly requested numeric storage; native uploads keep bytes."""
    if target_dtype not in ("float16", "float32"):
        raise ValueError("numeric weight conversion requires float16 or float32")
    if source_dtype == "F8_E4M3":
        from akas.quantization._shared.e4m3 import decode_storage

        if np.any(raw & 127 == 127):
            raise ValueError("resident FP8 storage must be finite")
        values = decode_storage(raw)
    elif source_dtype == "BF16":
        from tensor.runtime.dtypes import decode_bfloat16

        values = decode_bfloat16(raw.view("uint16"))
    elif source_dtype in ("F16", "F32"):
        values = raw.view(DTYPES[source_dtype])
    else:
        raise ValueError(f"unsupported numeric source storage {source_dtype!r}")
    return np.ascontiguousarray(values.astype(target_dtype)).view("uint8")


def upload_shards(device, tensors, wanted, layouts, *, conversions=None):
    """Map one shard at a time and copy bytes into declared grouped layouts."""
    conversions = dict(conversions or {})
    groups = {layouts[name][0] for name in wanted}
    if conversions.keys() - groups:
        raise ValueError("weight conversion names an undeclared storage group")
    group_dtypes = {}
    for name in wanted:
        group, offset, _ = layouts[name]
        dtype = tensors[name].dtype
        if group_dtypes.setdefault(group, dtype) != dtype:
            raise ValueError("grouped weight storage has inconsistent source dtypes")
        source_width = 2 if dtype in ("BF16", "F16") else 4 if dtype == "F32" else 1
        if group in conversions and (
            conversions[group] not in ("float16", "float32") or offset % source_width
        ):
            raise ValueError("invalid numeric weight conversion or group offset")
    buffers = {}
    try:
        for name in wanted:
            group, _, shape = layouts[name]
            if group not in buffers:
                buffers[group] = device.empty(
                    shape, conversions.get(group, DTYPES[tensors[name].dtype])
                )
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
                        data = view
                        destination_offset = offset
                        if group in conversions:
                            data = convert_storage(
                                view.copy(), info.dtype, conversions[group]
                            )
                            source_width = (
                                2
                                if info.dtype in ("BF16", "F16")
                                else 4
                                if info.dtype == "F32"
                                else 1
                            )
                            destination_offset = (
                                offset
                                // source_width
                                * np.dtype(conversions[group]).itemsize
                            )
                        device._check()
                        device.driver.call(
                            "cuMemcpyHtoD_v2",
                            buffers[group].pointer + destination_offset,
                            ct.c_void_p(data.ctypes.data),
                            data.nbytes,
                        )
                        device.driver.call("cuStreamSynchronize", None)
                    finally:
                        del data
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
