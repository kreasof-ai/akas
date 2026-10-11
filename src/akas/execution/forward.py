"""Compiler-free loading and ownership for composed forward bundles."""

import json
from pathlib import Path
from akas.artifacts.kernels import load_kernels
from akas.planning.forward import ForwardPlan


class Forward:
    def __init__(
        self, directory, device, buffers, *, filename="forward.json", graphs=False
    ):
        self.device, self.closed = device, False
        self.owned, self.kernels, self.graph = [], {}, None
        manifest = json.loads((Path(directory) / filename).read_text())
        if manifest.get("schema") != "akas.forward.v1":
            raise ValueError("unsupported forward bundle")
        self.definition = ForwardPlan.from_metadata(manifest["forward"])
        self.buffers = dict(buffers)
        try:
            if set(self.buffers) - set(self.definition.buffers):
                raise ValueError("unknown supplied buffer")
            for name, spec in self.definition.buffers.items():
                if name in self.buffers:
                    continue
                if spec.role not in ("scratch", "state"):
                    raise ValueError(f"missing supplied buffer: {name}")
                if spec.role == "state":
                    import numpy as np

                    value = device.from_numpy(
                        np.zeros(
                            spec.shape,
                            dtype="float32" if spec.dtype == "bfloat16" else spec.dtype,
                        ),
                        dtype=spec.dtype,
                    )
                else:
                    value = device.empty(spec.shape, spec.dtype)
                self.owned.append(value)
                self.buffers[name] = value
            self.kernels = load_kernels(
                device,
                directory,
                manifest["kernels"],
                requirements=self.definition.requirements(),
            )
            self.plan = self.definition.bind(device, self.kernels, self.buffers)
            if graphs:
                self.graph = self.plan.capture(
                    resources=tuple(self.buffers.values())
                    + tuple(self.kernels.values())
                )
        except BaseException:
            self.close()
            raise

    def run(self):
        if self.closed:
            raise RuntimeError("forward is closed")
        if self.graph is None:
            self.plan.submit()
        else:
            self.graph.launch()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.graph is not None:
            self.graph.close()
        for kernel in self.kernels.values():
            kernel.release()
        for buffer in reversed(self.owned):
            buffer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
