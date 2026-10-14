"""Request-slot ownership independent of model, storage format and device."""

import numpy as np


class StatePool:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.positions = np.zeros(capacity,dtype="int32")
        self.occupied = np.zeros(capacity,dtype=bool)
        self.generations = np.zeros(capacity, dtype="uint64")
        self.pending = {}

    def check(self, slot):
        if type(slot) is not int or not 0 <= slot < self.capacity or not self.occupied[slot]:
            raise ValueError("request is not owned")

    def reserve(self, slot, *, reset):
        if type(slot) is not int or not 0 <= slot < self.capacity or self.occupied[slot]:
            raise ValueError("request slot is unavailable")
        reset(slot)
        self.positions[slot] = 0
        self.occupied[slot] = True
        self.generations[slot] += 1
        return slot

    def free(self, slot, *, release):
        self.check(slot)
        if slot in self.pending:
            raise ValueError("discard pending verification before releasing its request")
        release(slot)
        self.occupied[slot] = False
        self.positions[slot] = 0

    def clear(self):
        self.generations += 1
        self.occupied.fill(False)
        self.positions.fill(0)
        self.pending.clear()


class Journal:
    """Operation-described delayed state sharing the program's buffer ownership."""

    def __init__(self, executor, frame, descriptor):
        import json
        from akas.graph import Graph
        from akas.execution.forward import Forward

        self.executor, self.frame = executor, frame
        self.dirty = next((journal.dirty for (rows,window),journal in executor.journals.items()
                           if window == frame.chunk), np.zeros(executor.capacity,dtype=bool))
        self.forwards = {}
        try:
            for phase in ("verify","commit","materialize"):
                record = descriptor[phase]
                definition = Graph.from_metadata(json.loads((executor.bundle/record["path"]).read_text())["forward"])
                available = {**frame.buffers,**executor.batches.states}
                if phase == "materialize":
                    available["mapping"] = frame.buffers[frame.inputs["mapping"]]
                supplied = {name:value for name,value in available.items()
                            if name in definition.buffers and definition.buffers[name].alias_of is None}
                forward = Forward(executor.bundle,executor.device,supplied,filename=record["path"],graphs=phase=="materialize")
                self.forwards[phase] = forward
                executor.batches._adopt_states(forward)
        except BaseException:
            self.close()
            raise

    def materialize(self, slots):
        slots = [slot for slot in slots if self.dirty[slot]]
        if not slots:
            return
        from akas.memory.transfer import write
        if len(slots) > self.frame.rows:
            for begin in range(0,len(slots),self.frame.rows):
                self.materialize(slots[begin:begin+self.frame.rows])
            return
        mapping = np.full(self.frame.rows,-1,dtype="int32")
        mapping[:len(slots)] = slots
        forward = self.forwards["materialize"]
        write(forward.device,forward.buffers["mapping"],mapping)
        forward.run()
        self.dirty[slots] = False

    def close(self):
        for forward in reversed(tuple(self.forwards.values())):
            forward.close()
        self.forwards.clear()
