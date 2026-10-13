"""Coordinate request scheduling over an explicit model adapter."""

from collections import deque
from threading import get_ident
from akas.config import EngineConfig
from akas.requests import Request
from akas.scheduling.scheduler import Scheduler


class Engine:
    """Owner-thread engine for greedy, length-bounded token-ID generation.

    The caller owns the Tensor device. The engine owns its model adapter;
    close the engine before closing that device. GPU batching is an adapter
    capability, so independent handles do not imply native batching.
    """

    def __init__(self, model, *, config=None, publish=None):
        self.thread = get_ident()
        self.model = model
        self.config = config or EngineConfig()
        self.events = deque()
        self.closed = False
        self.scheduler = Scheduler(
            model,
            publish if publish is not None else self.events.extend,
            prefill_chunk=self.config.prefill_chunk,
            prefill_requests=self.config.prefill_requests,
        )

    @classmethod
    def from_checkpoint(
        cls,
        definition,
        checkpoint,
        bundle,
        device,
        *,
        adapter=None,
        config=None,
        publish=None,
        **options,
    ):
        """Load caller-selected resources and optionally adapt their request API."""
        model = definition(checkpoint, bundle, device, **options)
        try:
            scheduled = model if adapter is None else adapter(model)
            return cls(scheduled, config=config, publish=publish)
        except BaseException:
            model.close()
            raise

    def submit(self, prompt, *, max_tokens):
        prompt = list(prompt)
        if not prompt or any(
            type(token) is not int or token < 0 or token >= self.model.config.vocab
            for token in prompt
        ):
            raise ValueError("prompt must contain valid integer token IDs")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if len(prompt) + max_tokens > self.model.context:
            raise ValueError("generation exceeds context capacity")
        request = Request(prompt, max_tokens)
        self.scheduler.submit(request)
        return request

    def _check_thread(self):
        if get_ident() != self.thread:
            raise RuntimeError("engine must run on its owner thread")

    def step(self):
        self._check_thread()
        if self.closed:
            raise RuntimeError("engine is closed")
        return self.scheduler.turn()

    def cancel(self, request):
        request.cancelled.set()

    def run(self):
        self._check_thread()
        try:
            self.scheduler.run()
        finally:
            self.close()

    def close(self):
        self._check_thread()
        if self.closed:
            return
        self.scheduler.stopped.set()
        try:
            for request in (*self.scheduler.prefilling, *self.scheduler.decoding):
                if request.slot is not None:
                    self.scheduler._finish(request, cancelled=True)
            self.scheduler.prefilling.clear()
            self.scheduler.decoding.clear()
            self.scheduler.pending.clear()
        finally:
            self.model.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
