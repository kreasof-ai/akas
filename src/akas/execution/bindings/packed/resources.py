"""Loaded LFM2 resources and independent request-handle lifecycle."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from akas.execution.workspace import DeviceOwner, release
from types import MappingProxyType
import json
import numpy as np
from akas.execution.bindings.packed.workspace import allocate_workspaces
from akas.artifacts.read import artifact_path
from akas.artifacts.read import load_kernels
from akas.artifacts.fingerprint import packed_hashes as implementation_hashes
from akas.ops.schedules.packed_parameters import CUDA_PROFILES
from akas.ops.schedules.packed_parameters import CUDA_GROUPED_THRESHOLD
from akas.ops.schedules.packed_selection import prefill_tail_rows
from akas.ops.schedules.packed_selection import valid_rows
from akas.execution.bindings.packed.requirements import requirements
from akas.execution.bindings.packed.request import LFM2Request


class PackedResources(DeviceOwner):
    """One weight/kernel allocation and serialized executor scratch space."""

    def __init__(
        self,
        model,
        bundle,
        device,
        *,
        context,
        max_requests,
        checkpoint_type,
        definition,
    ):
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        self.checkpoint_type = checkpoint_type
        self.definition = definition
        super().__init__(device)
        self.max_requests = max_requests
        self.requests = set()
        self.closed = False
        self.buffers = []
        self.kernels = {}
        try:
            self._load(model, bundle, device, context)
        except BaseException:
            self.close()
            raise

    def _load(self, model, bundle, device, context):
        self.checkpoint = self.checkpoint_type(model)
        self.gguf = self.checkpoint.gguf
        self.config = self.checkpoint.config
        self.architecture = self.definition(self.config, self.checkpoint)
        self.tokenizer = self.checkpoint.tokenizer
        self.device = device
        self.directory = Path(bundle)
        self.manifest = json.loads((self.directory / "inference.json").read_text())
        self.provider = device.info["provider"]
        if self.provider not in ("cuda", "webgpu"):
            raise ValueError("LFM2 requires CUDA or WebGPU")
        if self.manifest.get("provider", "cuda") != self.provider:
            raise ValueError("LFM2 bundle requires a different provider")
        if self.manifest.get("schema") != "tensor.lfm2-inference.v1":
            raise ValueError("unsupported LFM2 bundle schema")
        if self.manifest["config"] != json.loads(json.dumps(asdict(self.config))):
            raise ValueError("bundle belongs to a different LFM2 architecture")
        if self.manifest.get("implementation") != implementation_hashes(self.provider):
            raise ValueError(
                "LFM2 bundle implementation mismatch; rebuild with the installed producer"
            )
        self.capacity = self.manifest["capacity"]
        self.rows = tuple(self.manifest["rows"])
        self.cuda_profile = self.manifest.get("cuda_profile", "default")
        if self.cuda_profile not in CUDA_PROFILES:
            raise ValueError("unsupported CUDA kernel profile")
        self.cuda_optimized = (
            self.provider == "cuda" and self.cuda_profile == "optimized"
        )
        self.grouped_decode = self.cuda_optimized and (
            self.config.heads,
            self.config.kv_heads,
            self.config.head_dim,
        ) == (32, 8, 64)
        self.grouped_threshold = (
            self.manifest.get("schedule_profile", {})
            .get("runtime", {})
            .get("grouped_attention_threshold", CUDA_GROUPED_THRESHOLD)
        )
        if type(self.grouped_threshold) is not int or self.grouped_threshold < 1:
            raise ValueError("invalid grouped attention threshold")
        profile = (
            self.cuda_profile
            if self.provider == "cuda"
            else self.manifest.get("webgpu_profile", "portable")
        )
        if not valid_rows(self.provider, self.rows, profile) or self.capacity % 64:
            raise ValueError("unsupported LFM2 prefill/capacity profile")
        self.validate_context(context)
        self.webgpu_profile = self.manifest.get("webgpu_profile", "portable")
        # The final FFN feeds only logits; all persistent layer state is already
        # written by its mixer. Restrict this measured opt-in shape to one row.
        self.prefill_last = (
            self.provider == "webgpu"
            and self.webgpu_profile == "prefill_mixed"
            and (self.config.width, self.config.ff) == (2048, 10752)
            and all(
                self.gguf.tensors[
                    f"blk.{len(self.config.layers) - 1}.ffn_{name}.weight"
                ].type
                == 2
                for name in ("gate", "up", "down")
            )
        )
        if self.cuda_optimized:
            self.prefill_last = (
                self.gguf.tensors[
                    f"blk.{len(self.config.layers) - 1}.ffn_gate.weight"
                ].type
                == self.gguf.tensors[
                    f"blk.{len(self.config.layers) - 1}.ffn_up.weight"
                ].type
            )
        self.prefill_tail = (
            prefill_tail_rows(
                self.config,
                self.gguf,
                "cuda_optimized" if self.cuda_optimized else self.webgpu_profile,
            )
            if self.provider == "webgpu" or self.cuda_optimized
            else 0
        )
        expected = requirements(
            self.gguf,
            self.capacity,
            self.rows,
            provider=self.provider,
            webgpu_profile=self.webgpu_profile,
            cuda_profile=self.cuda_profile,
        )
        if set(expected) != set(self.manifest["kernels"]):
            raise ValueError("incomplete/incompatible LFM2 kernel coverage")
        for record in self.manifest["kernels"].values():
            artifact_path(self.directory, record)

        def upload(host):
            b = device.from_numpy(np.ascontiguousarray(host))
            self.buffers.append(b)
            return b

        self.weights = self.checkpoint.upload(device)
        self.buffers.extend(self.weights.values())
        self.weight_bytes = sum(b.nbytes for b in self.weights.values())
        self.weights = MappingProxyType(self.weights)
        allocate_workspaces(self, upload)
        self.kernels = load_kernels(device, self.directory, self.manifest["kernels"])

    def validate_context(self, context):
        if type(context) is not int or not 1 <= context <= min(
            self.capacity - max(self.rows), self.config.max_context
        ):
            raise ValueError("context exceeds the compiled capacity")

    def check(self):
        self._check()

    @property
    def allocated_bytes(self):
        return sum(b.nbytes for b in self.buffers if not b._released)

    def close(self):
        if self.closed:
            return
        self.check()
        self.device.synchronize()
        release(kernels=self.kernels.values(), buffers=self.buffers)
        self.closed = True


class PackedModel(LFM2Request):
    """Own a shared GGUF model and its backward-compatible default request.

    Set ``max_requests`` to opt into additional independent request handles.
    All handles share weights/kernels/executor scratch on one thread and stream.
    Closing the model closes every handle; closing a sibling releases only its
    request resources. Convolution is FP32 and attention caches are FP16.
    """

    def __init__(
        self, model, bundle, device, *, context=None, graphs=True, max_requests=1
    ):
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        if context is None:
            manifest = json.loads((Path(bundle) / "inference.json").read_text())
            context = min(
                manifest["capacity"] - max(manifest["rows"]),
                manifest["config"]["max_context"],
            )
        resources = PackedResources(
            model,
            bundle,
            device,
            context=context,
            max_requests=max_requests,
            checkpoint_type=self.checkpoint_type,
            definition=self.definition,
        )
        try:
            super().__init__(resources, context=context, graphs=graphs)
        except BaseException:
            resources.close()
            raise

    def new_request(self, *, context=None, graphs=None):
        """Allocate an independent sequence, counting the default toward capacity."""
        self._check_open()
        return LFM2Request(
            self._resources,
            context=self.context if context is None else context,
            graphs=self.graphs_enabled if graphs is None else graphs,
        )

    @property
    def shared_bytes(self):
        """Live weights and executor buffers, counted once across all requests."""
        return self._resources.allocated_bytes

    @property
    def weight_bytes(self):
        return self._resources.weight_bytes

    @property
    def request_count(self):
        return len(self._resources.requests)

    @property
    def allocated_bytes(self):
        """All live model/request buffers; excludes driver-owned graph storage."""
        return self.shared_bytes + sum(
            request.private_bytes for request in self._resources.requests
        )

    def close(self):
        resources = self._resources
        if resources.closed:
            return
        resources.check()
        for request in tuple(resources.requests):
            LFM2Request.close(request)
        resources.close()
