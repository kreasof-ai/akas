"""Allocate explicit persistent-state layouts with cleanup on partial failure."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BufferSpec:
    shape: tuple[int, ...]
    dtype: str
    zero: bool = False


def allocate_states(device, specifications):
    states = {}
    try:
        for layer, specs in specifications.items():
            buffers = states[layer] = []
            for spec in specs:
                buffer = (
                    device.from_numpy(np.zeros(spec.shape, dtype=spec.dtype))
                    if spec.zero
                    else device.empty(spec.shape, spec.dtype)
                )
                buffers.append(buffer)
        return states
    except BaseException:
        for buffers in states.values():
            for buffer in buffers:
                buffer.release()
        raise
