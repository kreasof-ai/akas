"""Compiler-free loading and ownership for composed forward bundles."""

import json
from pathlib import Path
from akas.artifacts.read import load_kernels
from akas.graph import Graph
from akas.execution.plan import bind_graph
from akas.execution.workspace import release
from akas.execution.workspace import Workspace


class Forward:
    def __init__(
        self, directory, device, buffers, *, filename="forward.json", graphs=False
    ):
        self.device, self.closed = device, False
        self.owned, self.kernels, self.graph = [], {}, None
        self.workspace = None
        manifest = json.loads((Path(directory) / filename).read_text())
        if manifest.get("schema") != "akas.forward.v1":
            raise ValueError("unsupported forward bundle")
        self.definition = Graph.from_metadata(manifest["forward"])
        self.buffers = dict(buffers)
        try:
            self.workspace = Workspace(self.definition, device, buffers, manifest.get("allocations"))
            self.buffers = self.workspace.buffers
            self.owned = self.workspace.owned
            self.kernels = load_kernels(
                device,
                directory,
                manifest["kernels"],
                requirements=self.definition.requirements(),
            )
            self.plan = bind_graph(self.definition, device, self.kernels, self.buffers)
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
        release(
            graphs=(self.graph,),
            kernels=self.kernels.values(),
        )
        if self.workspace is not None:
            self.workspace.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
