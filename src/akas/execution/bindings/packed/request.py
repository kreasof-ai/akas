"""Independent LFM2 request state and forward execution."""

from __future__ import annotations
import numpy as np
from akas.tensor.executor import bind
from akas.tensor.executor import capture
from akas.tensor.executor import submit
from akas.memory.layouts import allocate_hybrid_states
from akas.artifacts.schema import identity
from akas.execution.bindings.packed.plan import build_plan
from akas.generation.greedy import generate_greedy


_RESOURCE_FIELDS = (
    "architecture",
    "gguf",
    "config",
    "tokenizer",
    "device",
    "directory",
    "manifest",
    "provider",
    "capacity",
    "rows",
    "cuda_profile",
    "cuda_optimized",
    "grouped_decode",
    "grouped_threshold",
    "webgpu_profile",
    "prefill_last",
    "prefill_tail",
    "weights",
    "workspaces",
    "kernels",
)


class LFM2Request:
    """Independent sequence state sharing one model's weights and scratch.

    Obtain handles with ``LFM2.new_request``. Calls run on the model owner
    thread/stream and may be interleaved at call boundaries. This is not batching.
    """

    def __init__(self, resources, *, context, graphs):
        resources.check()
        resources.validate_context(context)
        if len(resources.requests) >= resources.max_requests:
            raise RuntimeError("LFM2 request capacity exhausted")
        self._resources = resources
        for name in _RESOURCE_FIELDS:
            setattr(self, name, getattr(resources, name))
        self.context = context
        self.graphs_enabled = graphs
        self.position = 0
        self.closed = False
        self.graphs = {}
        self.plans = {}
        self.buffers = []
        self.prepared = {}
        self.prepared_states = {}
        self.greedy = {}
        self.state_plans = {}
        self.state_graphs = {}
        self.greedy_plans = {}
        self.long_plan = None
        self.long_greedy_plan = None
        self.long_graph = None
        self.long_greedy_graph = None
        try:
            self._initialize(graphs)
        except BaseException:
            LFM2Request.close(self)
            raise
        resources.requests.add(self)

    def _initialize(self, graphs):
        device = self.device
        resources = self._resources

        def upload(host):
            b = device.from_numpy(np.ascontiguousarray(host))
            self.buffers.append(b)
            return b

        self.control = upload(np.array([0, 1], np.int32))
        self.logits = upload(np.zeros(self.config.vocab, np.float32))
        persistent = allocate_hybrid_states(
            device,
            self.config,
            1,
            self.capacity,
            kv_dtype="float16",
            zero_recurrent=True,
            zero_cache=True,
            flat=True,
        )
        self.states = {
            layer: values[0]
            for layer, values in persistent.items()
            if self.config.layers[layer] == "conv"
        }
        self.caches = {
            layer: tuple(values)
            for layer, values in persistent.items()
            if self.config.layers[layer] == "attention"
        }
        self.buffers.extend(
            buffer for values in persistent.values() for buffer in values
        )
        for r in self.rows:
            self.plans[r] = self._plan(r)
        if self.grouped_decode:
            self.long_plan = self._plan(1, grouped=True)
        if self.cuda_optimized:
            kinds = {
                id(self.kernels[key]): record["kind"]
                for key, record in self.manifest["kernels"].items()
            }
            for r, plan in self.plans.items():
                if r > 1:
                    end = max(
                        i
                        for i, (kernel, _) in enumerate(plan)
                        if kinds[id(kernel)] in ("conv", "qkv", "kvnorm")
                    )
                    self.state_plans[r] = (*plan[: end + 1], plan[-1])
            sample = self.kernels[identity("argmax", dict(n=self.config.vocab))]
            call = bind(
                device,
                sample,
                (self.logits, self.workspaces[1]["tokens"], self.control),
            )
            self.greedy_plans = {
                r: (*plan, (sample, call)) for r, plan in self.plans.items()
            }
            if self.long_plan:
                self.long_greedy_plan = (*self.long_plan, (sample, call))
        if self.provider == "webgpu":
            for r, plan in self.plans.items():
                self.prepared[r] = capture(device, plan, resources=())
            if self.prefill_tail:
                kinds = {
                    id(self.kernels[key]): record["kind"]
                    for key, record in self.manifest["kernels"].items()
                }
                for r, plan in self.plans.items():
                    if r == 1:
                        continue
                    end = max(
                        i
                        for i, (kernel, _) in enumerate(plan)
                        if kinds[id(kernel)] == "conv"
                    )
                    self.prepared_states[r] = capture(
                        device, (*plan[: end + 1], plan[-1]), resources=()
                    )
            sample = self.kernels[identity("argmax", dict(n=self.config.vocab))]
            call = bind(
                device,
                sample,
                (self.logits, self.workspaces[1]["tokens"], self.control),
            )
            for r, plan in self.plans.items():
                self.greedy[r] = capture(device, (*plan, (sample, call)), resources=())
            self.graphs_enabled = False
        elif graphs:
            for r, plan in self.plans.items():
                self.graphs[r] = capture(
                    device,
                    plan,
                    resources=(
                        *resources.buffers,
                        *self.buffers,
                        *self.kernels.values(),
                    ),
                )
            for r, plan in self.state_plans.items():
                self.state_graphs[r] = capture(
                    device,
                    plan,
                    resources=(
                        *resources.buffers,
                        *self.buffers,
                        *self.kernels.values(),
                    ),
                )
            for r, plan in self.greedy_plans.items():
                self.greedy[r] = capture(
                    device,
                    plan,
                    resources=(
                        *resources.buffers,
                        *self.buffers,
                        *self.kernels.values(),
                    ),
                )
            if self.long_plan:
                self.long_graph = capture(
                    device,
                    self.long_plan,
                    resources=(
                        *resources.buffers,
                        *self.buffers,
                        *self.kernels.values(),
                    ),
                )
                self.long_greedy_graph = capture(
                    device,
                    self.long_greedy_plan,
                    resources=(
                        *resources.buffers,
                        *self.buffers,
                        *self.kernels.values(),
                    ),
                )

    def _check_open(self):
        if self.closed:
            raise RuntimeError("LFM2 request is closed")
        self._resources.check()

    @property
    def private_bytes(self):
        """Live request buffers; excludes shared resources and driver graph storage."""
        return sum(b.nbytes for b in self.buffers if not b._released)

    @property
    def allocated_bytes(self):
        """Shared buffers plus this request; use the owner's total for many handles."""
        return self._resources.allocated_bytes + self.private_bytes

    def _plan(self, r, *, grouped=False):
        return build_plan(self, r, grouped=grouped)

    def _submit(self, plan):
        submit(self.device, plan)

    def _write(self, buffer, host):
        host = np.ascontiguousarray(host, dtype=buffer.dtype)
        if host.nbytes != buffer.nbytes:
            raise ValueError("host write shape mismatch")
        self.device._check()
        buffer._check()
        if self.provider == "webgpu":
            self.device.write(buffer, host.reshape(buffer.shape))
        else:
            self.device.driver.call(
                "cuMemcpyHtoD_v2", buffer.pointer, host.ctypes.data, host.nbytes
            )
            # Pageable copies may return after staging, before device DMA ends.
            # Graph execution uses a nonblocking stream and cannot implicitly
            # wait for this default-stream transfer (also required during reset).
            self.device.driver.call("cuStreamSynchronize", None)

    def reset(self):
        self._check_open()
        if self.provider != "webgpu":
            self.device.synchronize()
        self.position = 0
        self._write(self.control, np.array([0, 1], np.int32))
        for state in self.states.values():
            self._write(state, np.zeros(state.shape, np.float32))

    def forward(self, tokens, *, read=True, _greedy=False):
        self._check_open()
        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or not tokens.size or tokens.dtype.kind not in "iu":
            raise ValueError("requires nonempty integer token IDs")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab):
            raise ValueError("token ID outside vocabulary")
        if self.position + len(tokens) > self.context:
            raise ValueError("context capacity exceeded")
        result = None
        start = 0
        while start < len(tokens):
            remaining = len(tokens) - start
            prefill = [r for r in self.rows if r > 1]
            available = [r for r in prefill if r <= remaining]
            chunk = max(available) if available else min(prefill)
            values = tokens[start : start + chunk]
            r = 1 if len(values) == 1 else chunk
            host = np.zeros(r, np.int32)
            host[: len(values)] = values
            if self.provider != "webgpu":
                self.device.synchronize()
            self._write(self.workspaces[r]["tokens"], host)
            self._write(self.control, np.array([self.position, len(values)], np.int32))
            if self.provider == "webgpu":
                last = start + len(values) == len(tokens)
                output = self.workspaces[1]["tokens"] if _greedy else self.logits
                executable = (
                    self.prepared_states[r]
                    if not last and r in self.prepared_states
                    else (self.greedy if _greedy and last else self.prepared)[r]
                )
                result = executable.launch(readback=output if read and last else None)
            else:
                last = start + len(values) == len(tokens)
                mode = (
                    "state"
                    if not last and r in self.state_plans
                    else (
                        "greedy" if _greedy and last and self.cuda_optimized else "full"
                    )
                )
                plans = (
                    self.state_plans
                    if mode == "state"
                    else self.greedy_plans
                    if mode == "greedy"
                    else self.plans
                )
                graphs = (
                    self.state_graphs
                    if mode == "state"
                    else self.greedy
                    if mode == "greedy"
                    else self.graphs
                )
                if (
                    r == 1
                    and self.long_plan
                    and self.position + 1 >= self.grouped_threshold
                ):
                    if self.graphs_enabled:
                        (
                            self.long_greedy_graph
                            if mode == "greedy"
                            else self.long_graph
                        ).launch()
                    else:
                        self._submit(
                            self.long_greedy_plan
                            if mode == "greedy"
                            else self.long_plan
                        )
                elif self.graphs_enabled:
                    graphs[r].launch()
                else:
                    self._submit(plans[r])
            self.position += len(values)
            start += len(values)
        if self.provider == "webgpu":
            return result
        return (
            (
                self.workspaces[1]["tokens"]
                if _greedy and self.cuda_optimized
                else self.logits
            ).to_numpy()
            if read
            else None
        )

    def next_token(self, tokens):
        """Advance this sequence and return one greedy token ID."""
        if self.provider == "webgpu" or self.cuda_optimized:
            return int(self.forward(tokens, _greedy=True)[0])
        return int(np.argmax(self.forward(tokens)))

    def generate(self, prompt, *, max_tokens=128, chat=True, gpu_greedy=True):
        return generate_greedy(
            self, prompt, max_tokens=max_tokens, chat=chat, gpu_greedy=gpu_greedy
        )

    def close(self):
        if self.closed:
            return
        self._resources.check()
        self.device.synchronize()
        for graph in self.graphs.values():
            graph.close()
        for graph in self.state_graphs.values():
            graph.close()
        for graph in (self.long_graph, self.long_greedy_graph):
            if graph:
                graph.close()
        for plan in self.prepared.values():
            plan.close()
        for plan in self.prepared_states.values():
            plan.close()
        for plan in self.greedy.values():
            plan.close()
        for buffer in self.buffers:
            buffer.release()
        self.closed = True
        self._resources.requests.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
