"""Small mutable control/state transfers shared by execution and measurements."""

import ctypes as ct
import numpy as np


def write(device, buffer, values):
    buffer._check()
    if buffer.device is not device:
        raise ValueError("buffer belongs to another device")
    values = np.ascontiguousarray(values, dtype=str(buffer.dtype))
    if tuple(values.shape) != tuple(buffer.shape):
        raise ValueError("buffer write shape mismatch")
    if device.info["provider"] == "webgpu":
        device.write(buffer, values)
    else:
        # Tensor launches on a nonblocking stream; order mutable controls after
        # prior launches and complete the default-stream copy before replay.
        device.synchronize()
        device.driver.call(
            "cuMemcpyHtoD_v2",
            buffer.pointer,
            ct.c_void_p(values.ctypes.data),
            values.nbytes,
        )
        device.driver.call("cuStreamSynchronize", None)
