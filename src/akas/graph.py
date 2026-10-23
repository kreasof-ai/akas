"""One ordered operation graph shared by producers and runtime binding."""

from dataclasses import dataclass, field
from akas.artifacts.schema import identity


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "scratch"
    alias_of: str | None = None

    def view(self, shape):
        graph = self._graph
        return graph.view(f"{self.name}.view.{len(graph.buffers)}", self, shape)

    def __post_init__(self):
        if (
            not self.name
            or not self.shape
            or any(type(d) is not int or d <= 0 for d in self.shape)
        ):
            raise ValueError("buffer needs a name and positive static dimensions")
        if self.role not in ("input", "output", "weight", "state", "scratch", "journal"):
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


class Graph:
    """Models declare buffers and compose calls without opening a device.

    Buffer roles express ownership: weights/inputs/outputs are supplied by the
    caller, scratch/state may be allocated by the runtime. A contract names the
    operation ABI; provider and hardware selection are producer concerns.
    """

    def __init__(self, *, contract, phase="forward"):
        self.contract, self.phase = contract, phase
        self.buffers = {}
        self.calls = []
        self.request = None

    def bind_requests(self, *, inputs, prediction, reset=(), features=None, prefill_features=()):
        """Declare how token-phase fields bind to this graph's buffers."""
        if any(name not in self.buffers for name in (*inputs.values(), prediction, *reset, *(features or {}).values())):
            raise ValueError("request bindings reference an undeclared buffer")
        if len(set(prefill_features)) != len(prefill_features) or set(prefill_features) - set(features or {}):
            raise ValueError("prefill features must name distinct declared features")
        self.request = dict(inputs=dict(inputs), prediction=prediction, reset=list(reset))
        if prefill_features:
            self.request["prefill_features"] = list(prefill_features)
        if features:
            self.request["features"] = dict(features)
        return self

    def buffer(self, name, shape, dtype, *, role="scratch", alias_of=None):
        value = Buffer(name, tuple(shape), dtype, role, alias_of)
        if name in self.buffers:
            raise ValueError(f"duplicate buffer: {name}")
        if alias_of is not None:
            from math import prod

            base = self.buffers.get(alias_of)
            if base is None or base.dtype != dtype or prod(base.shape) != prod(value.shape):
                raise ValueError("view must reshape an existing buffer of the same dtype and size")
            if base.role != role:
                raise ValueError("view must preserve its base ownership")
        self.buffers[name] = value
        object.__setattr__(value, "_graph", self)
        return value

    def view(self, name, base, shape):
        if not isinstance(base, Buffer) or self.buffers.get(base.name) is not base:
            raise ValueError("view base belongs to another graph")
        return self.buffer(name, shape, base.dtype, role=base.role, alias_of=base.name)

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

    def lifetimes(self):
        """Inclusive call ranges, including uses through borrowed views."""
        uses = {}
        for index, call in enumerate(self.calls):
            for name in (*call.arguments, *call.bindings.values()):
                while name is not None:
                    uses.setdefault(name, []).append(index)
                    name = self.buffers[name].alias_of
        if self.request:
            exposed = [self.request["prediction"], *self.request.get("features", {}).values()]
            for name in exposed:
                while name is not None:
                    uses.setdefault(name, []).append(len(self.calls))
                    name = self.buffers[name].alias_of
        return {name: (min(indices), max(indices)) for name, indices in uses.items()}

    def metadata(self):
        from dataclasses import asdict

        result = dict(
            contract=self.contract,
            phase=self.phase,
            buffers=[{k: v for k, v in asdict(value).items() if k != "alias_of" or v is not None}
                     for value in self.buffers.values()],
            calls=[asdict(call) for call in self.calls],
        )
        if self.request is not None:
            result["request"] = self.request
        return result

    @classmethod
    def from_metadata(cls, data):
        plan = cls(contract=data["contract"], phase=data["phase"])
        for row in data["buffers"]:
            plan.buffer(row["name"], row["shape"], row["dtype"], role=row["role"], alias_of=row.get("alias_of"))
        for row in data["calls"]:
            plan(
                row["kind"],
                row["parameters"],
                *row["arguments"],
                label=row.get("label"),
                **row["bindings"],
            )
        if data.get("request") is not None:
            plan.bind_requests(**data["request"])
        return plan
