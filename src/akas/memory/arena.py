"""Aligned CUDA slices backed by one allocation."""

import numpy as np


class UploadArena:
    """Retain one root until all slices close, including partial upload failure."""

    def __init__(self, device, sizes):
        if device.info.get("provider") != "cuda":
            raise ValueError("weight arenas require CUDA")
        self.device = device
        self.capacity = sum((n + 255) // 256 * 256 for n in sizes)
        self.root = device.empty((self.capacity,), "uint8")
        self.offset, self.references, self.sealed = 0, 0, False

    def empty(self, shape, dtype, *, zero=False):
        from tensor.runtime.buffer import Buffer
        from tensor.runtime.dtypes import dtype as normalize_dtype
        from math import prod

        if self.sealed:
            raise RuntimeError("weight arena upload is closed")
        dtype = normalize_dtype(dtype)
        nbytes = prod(shape) * dtype.itemsize
        if self.offset + nbytes > self.capacity:
            raise ValueError("weight arena capacity exceeded")
        arena = self

        class Lease:
            def __init__(self):
                self.active = True
                arena.references += 1

            def release(self):
                if self.active:
                    self.active = False
                    arena.references -= 1
                    arena._reclaim()

        result = Buffer(
            self.device,
            self.root.pointer + self.offset,
            tuple(shape),
            dtype,
            owner=Lease(),
        )
        result._base = self.root
        self.device._buffers.add(result)
        try:
            if zero:
                self.device.driver.call(
                    "cuMemsetD8Async",
                    result.pointer,
                    0,
                    result.nbytes,
                    self.device.stream,
                )
        except BaseException:
            result.release()
            raise
        self.offset += (nbytes + 255) // 256 * 256
        return result

    def upload(self, values):
        from akas.memory.transfer import write

        values = np.ascontiguousarray(values)
        result = self.empty(values.shape, values.dtype)
        try:
            write(self.device, result, values)
            return result
        except BaseException:
            result.release()
            raise

    def _reclaim(self):
        if self.sealed and self.references == 0 and self.root is not None:
            self.root.release()
            self.root = None

    def close(self):
        self.sealed = True
        self._reclaim()
