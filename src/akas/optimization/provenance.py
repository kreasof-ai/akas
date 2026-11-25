"""Identify dependencies that affect deployment compatibility."""

import importlib.metadata

from akas.optimization.context import identity


def software():
    result = {}
    for name in ("tensor-workspace", "tilelang", "apache-tvm-ffi", "wgpu", "numpy"):
        try:
            d = importlib.metadata.distribution(name)
            result[name] = dict(
                version=d.version, record=identity(d.read_text("RECORD"))
            )
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
