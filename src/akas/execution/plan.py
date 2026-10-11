"""Ordered execution plans borrow kernels and buffers owned by the engine."""

from akas.tensor.executor import bind
from akas.tensor.executor import submit
from akas.tensor.executor import capture


def call_bindings(kernel, call):
    """Recover buffer names from the compiled ABI, which may reorder arguments."""
    abi = kernel.manifest.get("abi", kernel.manifest["arguments"])
    return {item["name"]: value for item, value in zip(abi, call.storage)}


def bind_named(device, kernel, values):
    """Create a plan entry in the replacement kernel's declared argument order."""
    arguments = tuple(values[item["name"]] for item in kernel.manifest["arguments"])
    return kernel, bind(device, kernel, arguments)


class ExecutionPlan(list):
    def __init__(self, device, calls=(), *, labeled=False):
        super().__init__(calls)
        self.device = device
        self.labeled = labeled

    def bind(self, kernel, arguments, *, label=None):
        call = bind(self.device, kernel, arguments)
        self.append((kernel, call, label) if self.labeled else (kernel, call))
        return call

    def submit(self):
        submit(self.device, self)

    def capture(self, *, resources):
        return capture(self.device, self, resources=resources)


class KernelPlan:
    """Resolve kernel specializations and bind ordered or named operation calls.

    Plans and kernels are borrowed. A profile can transform specialization
    parameters, and a scratch cache can observe writes during composition.
    """

    def __init__(self, plan, kernels, *, identity=None, parameters=None, on_write=None):
        if identity is None:
            from akas.artifacts.kernels import identity
        self.plan = plan
        self.kernels = kernels
        self.identity = identity
        self.parameters = parameters
        self.on_write = on_write

    def __call__(self, kind, parameters, *arguments, label=None, **bindings):
        if arguments and bindings:
            raise TypeError("operation arguments must be positional or named")
        if self.parameters is not None:
            parameters = self.parameters(kind, parameters)
        from akas.planning.forward import ForwardPlan
        if isinstance(self.plan, ForwardPlan):
            self.plan(kind, parameters, *arguments, label=label, **bindings)
            if self.on_write is not None and "out" in bindings:
                self.on_write(bindings["out"])
            return
        kernel = self.kernels[self.identity(kind, parameters)]
        if bindings:
            arguments = tuple(
                bindings[item["name"]] for item in kernel.manifest["arguments"]
            )
        self.plan.bind(kernel, arguments, label=label)
        if self.on_write is not None and "out" in bindings:
            self.on_write(bindings["out"])
