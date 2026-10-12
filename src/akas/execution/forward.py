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
            from akas.memory.lifetimes import scratch_slots

            for names in scratch_slots(self.definition):
                # Caller-supplied scratch has explicit ownership and cannot be
                # reused by another name.
                names = [name for name in names if name not in self.buffers]
                if not names:
                    continue
                spec = self.definition.buffers[names[0]]
                value = device.empty(spec.shape, spec.dtype)
                self.owned.append(value)
                self.buffers.update((name, value) for name in names)
            for name, spec in self.definition.buffers.items():
                if name in self.buffers:
                    continue
                if spec.role not in ("scratch", "state"):
                    raise ValueError(f"missing supplied buffer: {name}")
                if spec.role == "state":
                    import numpy as np

                    host = np.zeros(
                        spec.shape,
                        dtype="float32" if spec.dtype == "bfloat16" else spec.dtype,
                    )
                    value = (
                        device.from_numpy(host, dtype=spec.dtype)
                        if spec.dtype == "bfloat16"
                        else device.from_numpy(host)
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
