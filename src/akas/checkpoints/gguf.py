"""Bounded, memory-mapped GGUF v3 reader and GGML block dequantization.

Tensor dimensions in the file have their contiguous dimension first; exposed
NumPy shapes reverse that order. Packed weights remain bytes until explicitly
dequantized or transferred to a device. No GGML/llama runtime is imported.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import struct
import numpy as np


class GGUFError(ValueError):
    pass


# GGML type: (name, values per block, bytes per block).
from akas.quantization.weights.ggml.definition import ENCODINGS as TYPES

SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    type: int
    offset: int
    nbytes: int

    @property
    def shape(self):
        return self.dimensions[::-1]

    @property
    def encoding(self):
        return TYPES[self.type][0]


class GGUF:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = {}
        self.tensors = {}
        size = self.path.stat().st_size
        with self.path.open("rb") as stream:

            def read(n):
                if n < 0 or n > size - stream.tell():
                    raise GGUFError("truncated GGUF field")
                value = stream.read(n)
                if len(value) != n:
                    raise GGUFError("truncated GGUF field")
                return value

            def scalar(fmt):
                return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

            def string():
                try:
                    return read(scalar("Q")).decode("utf-8")
                except UnicodeDecodeError as error:
                    raise GGUFError("invalid UTF-8 in GGUF") from error

            def value(kind):
                if kind == 8:
                    return string()
                if kind == 9:
                    subtype, count = scalar("I"), scalar("Q")
                    if subtype == 9 or subtype not in {*SCALARS, 8} or count > size:
                        raise GGUFError("invalid GGUF metadata array")
                    return [value(subtype) for _ in range(count)]
                if kind not in SCALARS:
                    raise GGUFError(f"unknown metadata type {kind}")
                return scalar(SCALARS[kind])

            if read(4) != b"GGUF" or scalar("I") != 3:
                raise GGUFError("requires little-endian GGUF version 3")
            tensor_count, metadata_count = scalar("Q"), scalar("Q")
            if max(tensor_count, metadata_count) > size // 12:
                raise GGUFError("invalid GGUF field count")
            for _ in range(metadata_count):
                name = string()
                if name in self.metadata:
                    raise GGUFError("duplicate GGUF metadata key")
                self.metadata[name] = value(scalar("I"))
            pending = []
            for _ in range(tensor_count):
                name, rank = string(), scalar("I")
                if name in self.tensors or not 1 <= rank <= 4:
                    raise GGUFError("invalid tensor name/rank")
                dims = tuple(scalar("Q") for _ in range(rank))
                kind, offset = scalar("I"), scalar("Q")
                if kind not in TYPES:
                    raise GGUFError(f"{name}: unsupported GGML type {kind}")
                _, block, bytes_per_block = TYPES[kind]
                if any(d < 1 for d in dims) or dims[0] % block:
                    raise GGUFError(f"{name}: invalid block dimensions")
                info = TensorInfo(
                    name, dims, kind, offset, math.prod(dims) // block * bytes_per_block
                )
                self.tensors[name] = info
                pending.append(info)
            alignment = self.metadata.get("general.alignment", 32)
            if type(alignment) != int or alignment < 1 or alignment & (alignment - 1):
                raise GGUFError("invalid GGUF alignment")
            self.data_offset = (stream.tell() + alignment - 1) // alignment * alignment
            end = 0
            for info in sorted(pending, key=lambda t: t.offset):
                if (
                    info.offset % alignment
                    or info.offset < end
                    or self.data_offset + info.offset + info.nbytes > size
                ):
                    raise GGUFError(f"{info.name}: invalid or overlapping tensor range")
                end = info.offset + info.nbytes
        self._bytes = np.memmap(self.path, mode="r", dtype=np.uint8)

    def packed(self, name):
        info = self.tensors[name]
        begin = self.data_offset + info.offset
        return self._bytes[begin : begin + info.nbytes]

    def array(self, name, *, dtype=np.float32):
        info = self.tensors[name]
        return (
            dequantize(self.packed(name), info.type)
            .reshape(info.shape)
            .astype(dtype, copy=False)
        )

    def row(self, name, index):
        info = self.tensors[name]
        if len(info.shape) != 2 or type(index) != int or not 0 <= index < info.shape[0]:
            raise GGUFError("invalid tensor row")
        count = info.nbytes // info.shape[0]
        return dequantize(
            self.packed(name)[index * count : (index + 1) * count], info.type
        )


def prepack_q4_0(data):
    """Compatibility boundary; numerical storage conversion belongs to GGML."""
    from akas.quantization.weights.ggml.kernel import prepack_host

    try:
        return prepack_host(data)
    except ValueError as error:
        raise GGUFError(str(error)) from error


def dequantize(data, kind):
    """Compatibility boundary; numerical storage conversion belongs to GGML."""
    from akas.quantization.weights.ggml.kernel import decode_host

    try:
        return decode_host(data, kind)
    except ValueError as error:
        raise GGUFError(str(error)) from error
