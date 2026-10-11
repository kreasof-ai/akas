"""One ordered operation graph shared by producers and runtime binding."""

from dataclasses import dataclass, field
from akas.artifacts.kernels import identity


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "scratch"

    def __post_init__(self):
        if (
            not self.name
            or not self.shape
            or any(type(d) is not int or d <= 0 for d in self.shape)
        ):
            raise ValueError("buffer needs a name and positive static dimensions")
        if self.role not in ("input", "output", "weight", "state", "scratch"):
            raise ValueError("invalid buffer role")


@dataclass(frozen=True)
class Call:
    kind: str
    parameters: dict
    arguments: tuple[str, ...] = ()
    bindings: dict[str, str] = field(default_factory=dict)
    label: object = None

    @property
    def key(self):
        return identity(self.kind, self.parameters)


class ForwardPlan:
    """Models declare buffers and compose calls without opening a device.

    Buffer roles express ownership: weights/inputs/outputs are supplied by the
    caller, scratch/state may be allocated by the runtime. A contract names the
    operation ABI; provider and hardware selection are producer concerns.
    """

    def __init__(self, *, contract, phase="forward"):
        self.contract, self.phase = contract, phase
        self.buffers = {}
        self.calls = []

    def buffer(self, name, shape, dtype, *, role="scratch"):
        value = Buffer(name, tuple(shape), dtype, role)
        if name in self.buffers:
            raise ValueError(f"duplicate buffer: {name}")
        self.buffers[name] = value
        return value

    def __call__(self, kind, parameters, *arguments, label=None, **bindings):
        if arguments and bindings:
            raise TypeError("operation arguments must be positional or named")

        def reference(value):
            name = value.name if isinstance(value, Buffer) else value
            if not isinstance(name, str) or name not in self.buffers:
                raise ValueError(f"undeclared buffer: {name}")
            return name

        call = Call(
            kind,
            dict(parameters),
            tuple(map(reference, arguments)),
            {name: reference(value) for name, value in bindings.items()},
            label,
        )
        self.calls.append(call)
        return call

    def requirements(self):
        return {call.key: (call.kind, call.parameters) for call in self.calls}

    def metadata(self):
        from dataclasses import asdict

        return dict(
            contract=self.contract,
            phase=self.phase,
            buffers=[asdict(value) for value in self.buffers.values()],
            calls=[asdict(call) for call in self.calls],
        )

    @classmethod
    def from_metadata(cls, data):
        plan = cls(contract=data["contract"], phase=data["phase"])
        for row in data["buffers"]:
            plan.buffer(row["name"], row["shape"], row["dtype"], role=row["role"])
        for row in data["calls"]:
            plan(
                row["kind"],
                row["parameters"],
                *row["arguments"],
                label=row.get("label"),
                **row["bindings"],
            )
        return plan

    def bind(self, device, kernels, buffers):
        from akas.execution.plan import ExecutionPlan, KernelPlan

        if set(buffers) != set(self.buffers):
            raise ValueError("forward buffer coverage mismatch")
        for name, spec in self.buffers.items():
            value = buffers[name]
            if tuple(value.shape) != spec.shape or str(value.dtype) != spec.dtype:
                raise ValueError(f"incompatible forward buffer: {name}")
        result = ExecutionPlan(device, labeled=True)
        emit = KernelPlan(result, kernels)
        for call in self.calls:
            emit(
                call.kind,
                call.parameters,
                *(buffers[name] for name in call.arguments),
                label=call.label,
                **{name: buffers[ref] for name, ref in call.bindings.items()},
            )
        return result
