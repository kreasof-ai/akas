"""Optional CUDA events around baseline execution; no scheduler modifications."""

import functools, importlib.abc, importlib.machinery, json, os, time
from pathlib import Path

_pending = []
_computed = {}


def control():
    try:
        return json.loads(Path(os.environ["AKAS_GPU_EVENT_CONTROL"]).read_text())
    except (OSError, KeyError, ValueError):
        return None


def write(row):
    root = Path(os.environ["AKAS_GPU_EVENT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"events-{os.getpid()}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def drain():
    global _pending
    keep = []
    for begin, end, meta in _pending:
        if end.query():
            write({**meta, "gpu_seconds": begin.elapsed_time(end) / 1000})
        else:
            keep.append((begin, end, meta))
    _pending = keep


def timed(callback, meta):
    import torch

    drain()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    stream = torch.cuda.current_stream()
    begin.record(stream)
    try:
        return callback()
    finally:
        end.record(stream)
        _pending.append((begin, end, {**meta, "stream": stream.cuda_stream}))


def patch(module):
    name = module.__name__
    if name.startswith("vllm."):
        cls = module.GPUModelRunner
        old = cls.execute_model

        @functools.wraps(old)
        def execute(self, scheduler_output, *args, **kwargs):
            ctx = control()
            scheduled = dict(scheduler_output.num_scheduled_tokens)
            if not ctx or not scheduled or kwargs.get("dummy_run", False):
                return old(self, scheduler_output, *args, **kwargs)
            prefix = sum(
                min(n, max(0, ctx["prefix"] - _computed.get(rid, 0)))
                for rid, n in scheduled.items()
            )
            meta = {
                **ctx,
                "num_requests": len(scheduled),
                "num_tokens": sum(scheduled.values()),
                "prefix_tokens": prefix,
                "pure_decode": prefix == 0 and all(n == 1 for n in scheduled.values()),
                "runner": name,
            }
            for rid, n in scheduled.items():
                _computed[rid] = _computed.get(rid, 0) + n
            self._akas_gpu_meta = meta
            return timed(
                lambda: old(self, scheduler_output, *args, **kwargs),
                {**meta, "stage": "forward"},
            )

        cls.execute_model = execute
        old_sample = cls.sample_tokens

        @functools.wraps(old_sample)
        def sample(self, *args, **kwargs):
            meta = getattr(self, "_akas_gpu_meta", None)
            if meta is None:
                return old_sample(self, *args, **kwargs)
            try:
                return timed(
                    lambda: old_sample(self, *args, **kwargs),
                    {**meta, "stage": "sampling"},
                )
            finally:
                self._akas_gpu_meta = None

        cls.sample_tokens = sample
    else:
        cls = module.TpModelWorker
        old = cls.forward_batch_generation

        @functools.wraps(old)
        def forward(self, batch, *args, **kwargs):
            ctx = control()
            if not ctx or batch is None:
                return old(self, batch, *args, **kwargs)
            decode = bool(batch.forward_mode.is_decode())
            n = len(batch.reqs)
            tokens = int(batch.input_ids.numel())
            meta = {
                **ctx,
                "num_requests": n,
                "num_tokens": tokens,
                "prefix_tokens": 0 if decode else tokens,
                "pure_decode": decode,
                "runner": name,
                "stage": "forward_sampling",
            }
            return timed(lambda: old(self, batch, *args, **kwargs), meta)

        cls.forward_batch_generation = forward
    write({"type": "hook_installed", "module": name})


class Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        patch(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    targets = {
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
        "sglang.srt.managers.tp_worker",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


def install():
    import sys

    if not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
