"""Allocation ownership and artifact-declared scratch/state/view bindings."""

from threading import get_ident


class DeviceOwner:
    def __init__(self, device=None):
        self.device = device
        self.thread = get_ident()
        self.generation = device._generation if device is not None else None
        self.closed = False

    def _check(self):
        if self.closed:
            raise RuntimeError("execution is closed")
        if get_ident() != self.thread:
            raise RuntimeError("execution must run on its owner thread")
        if self.device is not None:
            self.device._check()
            if self.generation != self.device._generation:
                raise RuntimeError("execution belongs to an expired device session")


def release(*, graphs=(), kernels=(), buffers=()):
    """Release explicitly owned resources, graphs before referenced allocations."""
    seen = set()
    for collection, method in (
        (graphs, "close"),
        (kernels, "release"),
        (buffers, "release"),
    ):
        for value in collection:
            if value is not None and id(value) not in seen:
                getattr(value, method)()
                seen.add(id(value))



class ScratchPool:
    """Stable scratch addresses shared by phases on one ordered device stream.

    Each phase numbers its allocation groups by decreasing byte size. A slot
    reserves the largest requirement at that rank across the compiled phases.
    Outputs and persistent state never borrow these allocations.
    """

    def __init__(self, device, capacities):
        self.device, self.capacities = device, tuple(capacities)
        self.buffers = {}

    def borrow(self, slot, spec):
        from math import prod
        from tensor.runtime.buffer import Buffer
        from tensor.runtime.dtypes import dtype

        dt = dtype(spec.dtype)
        nbytes = prod(spec.shape) * dt.itemsize
        if slot >= len(self.capacities) or nbytes > self.capacities[slot]:
            raise ValueError("phase exceeds its compiled scratch pool")
        if slot not in self.buffers:
            self.buffers[slot] = self.device.empty((self.capacities[slot],), "uint8")
        root = self.buffers[slot]

        class Lease:
            # Buffer disposal releases this descriptor; the bank owns root.
            def __init__(self, base):
                self.base = base

            def release(self):
                self.base = None

        value = Buffer(self.device, root.pointer, spec.shape, dt, owner=Lease(root))
        value._base = root
        self.device._buffers.add(value)
        return value

    def close(self):
        release(buffers=reversed(tuple(self.buffers.values())))
        self.buffers.clear()


class Workspace:
    def __init__(self, graph, device, supplied, allocations, *, scratch_pool=None):
        self.device, self.buffers = device, dict(supplied)
        self.owned, self.views, self.closed = [], [], False
        if set(supplied) - set(graph.buffers):
            raise ValueError("unknown supplied buffer")
        groups = allocations if allocations is not None else [
            [name] for name,spec in graph.buffers.items()
            if spec.role == "scratch" and spec.alias_of is None
        ]
        try:
            seen = set()
            lifetimes = graph.lifetimes()
            if scratch_pool is not None:
                from math import prod
                from tensor.runtime.dtypes import dtype

                groups = sorted(groups, key=lambda group: prod(graph.buffers[group[0]].shape)
                                * dtype(graph.buffers[group[0]].dtype).itemsize if group else 0,
                                reverse=True)
            for slot, group in enumerate(groups):
                if not group:
                    raise ValueError("empty scratch allocation group")
                for name in group:
                    spec = graph.buffers.get(name)
                    if spec is None or spec.role != "scratch" or spec.alias_of is not None or name in seen:
                        raise ValueError("invalid scratch allocation plan")
                    seen.add(name)
                intervals = sorted(lifetimes[name] for name in group if name in lifetimes)
                if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
                    raise ValueError("scratch allocations have overlapping lifetimes")
                names = [name for name in group if name not in self.buffers]
                if not names:
                    continue
                spec = graph.buffers[names[0]]
                if any((graph.buffers[name].shape,graph.buffers[name].dtype) != (spec.shape,spec.dtype) for name in names):
                    raise ValueError("scratch allocation shape/dtype mismatch")
                value = (scratch_pool.borrow(slot, spec) if scratch_pool is not None
                         else device.empty(spec.shape,spec.dtype))
                self.owned.append(value)
                self.buffers.update((name,value) for name in names)
            for name,spec in graph.buffers.items():
                if spec.alias_of is not None:
                    if name in supplied:
                        raise ValueError("views must bind from their declared base")
                    value = self.buffers[spec.alias_of].view(spec.shape)
                    self.views.append(value)
                    self.buffers[name] = value
                elif name not in self.buffers:
                    if spec.role not in ("scratch","state","journal"):
                        raise ValueError(f"missing supplied buffer: {name}")
                    if spec.role == "state":
                        value = device.zeros(spec.shape, spec.dtype)
                    else:
                        value = device.empty(spec.shape,spec.dtype)
                    self.owned.append(value)
                    self.buffers[name] = value
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        release(buffers=(*reversed(self.views), *reversed(self.owned)))
