"""CUDA-accessible pinned host storage for infrequently read weight rows."""

import ctypes as ct
import numpy as np


def from_numpy(device, values):
    """Copy encoded bytes to pinned host memory, without a device-side copy."""
    from tensor.runtime.buffer import Buffer
    from tensor.runtime.dtypes import storage_dtype

    if device.info.get("provider") != "cuda":
        raise ValueError("mapped host weights require CUDA")
    device._check()
    values = np.ascontiguousarray(values)
    nbytes = values.nbytes
    lib = device.driver.lib
    lib.cuMemHostAlloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_uint]
    lib.cuMemHostGetDevicePointer_v2.argtypes = [
        ct.POINTER(ct.c_uint64),
        ct.c_void_p,
        ct.c_uint,
    ]
    lib.cuMemFreeHost.argtypes = [ct.c_void_p]

    class Owner:
        def __init__(self):
            self.pointer = ct.c_void_p()
            self.closed = False
            device.driver.call("cuMemHostAlloc", ct.byref(self.pointer), nbytes, 2)

        def release(self):
            if not self.closed:
                device.driver.call("cuMemFreeHost", self.pointer)
                self.closed = True

    class MappedBuffer(Buffer):
        memory_kind = "mapped_host"

        def to_bytes(self):
            self._check()
            device.synchronize()
            return ct.string_at(self._owner.pointer, self.nbytes)

        def to_numpy(self):
            result = (
                np.frombuffer(self.to_bytes(), dtype=storage_dtype(self.dtype))
                .copy()
                .reshape(self.shape)
            )
            if str(self.dtype) == "bfloat16":
                from tensor.runtime.dtypes import decode_bfloat16

                return decode_bfloat16(result)
            return result

        def __dlpack_device__(self):
            return 3, device.ordinal

        def __dlpack__(self, **kwargs):
            raise BufferError("mapped host weight storage does not export DLPack")

    owner = Owner()
    try:
        pointer = ct.c_uint64()
        device.driver.call(
            "cuMemHostGetDevicePointer_v2", ct.byref(pointer), owner.pointer, 0
        )
        ct.memmove(owner.pointer, values.ctypes.data, values.nbytes)
        result = MappedBuffer(
            device, pointer.value, values.shape, values.dtype, owner=owner
        )
        device._buffers.add(result)
        return result
    except BaseException:
        owner.release()
        raise
