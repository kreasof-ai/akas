"""Bind independent sequence handles to the shared executor's request slots."""

from akas.execution.executor import Executor
from akas.models.contracts import Capabilities
from akas.models.contracts import IndependentModel


class HandleExecutor(Executor):
    capabilities = Capabilities(native_batch=False)

    def __init__(self, model: IndependentModel, *, capacity=1):
        super().__init__(
            capacity=capacity,
            context=model.context,
            device=getattr(model, "device", None),
        )
        self.model = model
        self.config = model.config
        self.handles = {}

    def _admit_slot(self, slot):
        request = self.model if slot == 0 else self.model.new_request()
        try:
            request.reset()
        except BaseException:
            if request is not self.model:
                request.close()
            raise
        self.handles[slot] = request

    def _release_slot(self, slot):
        request = self.handles[slot]
        request.reset() if request is self.model else request.close()
        del self.handles[slot]

    def _execute_batch(self, slots, sequences, chunk):
        predicted = []
        for slot, sequence in zip(slots, sequences):
            predicted.append(self.handles[slot].next_token(sequence))
            self.positions[slot] += len(sequence)
        return predicted

    def _close_resources(self):
        self.model.close()
        self.handles.clear()
