"""Schedule independent request handles without assuming a model or provider."""

from akas.models.contracts import Capabilities
from akas.models.contracts import IndependentModel


class IndependentAdapter:
    """Own a model whose handles share resources but execute one sequence at a time.

    Slot zero reuses the model's default request. Other slots allocate handles
    on admission and close them on release. Closing the model closes all handles.
    """

    capabilities = Capabilities(native_batch=False)

    def __init__(self, model: IndependentModel, *, capacity=1):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.model = model
        self.capacity = capacity
        self.context = model.context
        self.config = model.config
        self.handles = {}
        self.closed = False

    def _check(self):
        if self.closed:
            raise RuntimeError("independent adapter is closed")

    def admit(self):
        self._check()
        slot = next((i for i in range(self.capacity) if i not in self.handles), None)
        if slot is None:
            raise ValueError("request capacity exhausted")
        request = self.model if slot == 0 else self.model.new_request()
        try:
            request.reset()
        except BaseException:
            if request is not self.model:
                request.close()
            raise
        self.handles[slot] = request
        return slot

    def release(self, slot):
        self._check()
        request = self.handles[slot]
        request.reset() if request is self.model else request.close()
        del self.handles[slot]

    def prefill_batch(self, requests, sequences, chunk):
        self._check()
        return [
            self.handles[request.slot].next_token(tokens)
            for request, tokens in zip(requests, sequences)
        ]

    def decode(self, slots, tokens):
        self._check()
        return [
            self.handles[slot].next_token([token]) for slot, token in zip(slots, tokens)
        ]

    def close(self):
        if not self.closed:
            self.model.close()
            self.handles.clear()
            self.closed = True
