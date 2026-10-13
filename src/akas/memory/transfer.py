"""Small mutable control/state transfers shared by execution and measurements."""

import ctypes as ct
import numpy as np


def write(device, buffer, values):
    write_many(device, ((buffer, values),))


def write_many(device, updates):
    """Validate controls together and order one upload batch before replay."""
    prepared = []
    for buffer, values in updates:
        buffer._check()
        if buffer.device is not device:
            raise ValueError("buffer belongs to another device")
        if str(buffer.dtype) == "bfloat16":
            from tensor.runtime.dtypes import encode_bfloat16
            values = encode_bfloat16(values)
        else:
            values = np.ascontiguousarray(values, dtype=str(buffer.dtype))
        if tuple(values.shape) != tuple(buffer.shape):
            raise ValueError("buffer write shape mismatch")
        prepared.append((buffer, values))
    if not prepared:
        return
    if device.info["provider"] == "webgpu":
        for buffer, values in prepared:
            device.write(buffer, values)
    else:
        # Tensor launches on a nonblocking stream; order mutable controls after
        # prior launches and complete the default-stream copy before replay.
        device.synchronize()
        for buffer, values in prepared:
            device.driver.call("cuMemcpyHtoD_v2", buffer.pointer,
                               ct.c_void_p(values.ctypes.data), values.nbytes)
        device.driver.call("cuStreamSynchronize", None)
