"""Model declarations open the same compiled program and request executor."""

from dataclasses import dataclass
from importlib import import_module

from akas.execution.executor import Executor
from akas.graph import Graph


@dataclass(frozen=True)
class Composition:
    """The graph to execute and an optional accepted-prefix commit graph."""

    graph: Graph
    commit: Graph | None = None

    def __post_init__(self):
        if not isinstance(self.graph, Graph):
            raise TypeError("compose must supply a Graph")
        if self.commit is not None and not isinstance(self.commit, Graph):
            raise TypeError("compose commit must be a Graph or None")


class Definition:
    """Declare a checkpoint reader and a pure geometry-to-graph composition.

    Construction opens an already compiled program. Device allocation, phase
    selection and request state belong to Executor, never to a model subclass.
    """

    compilation_profile = "program"
    checkpoint_architectures = ()
    optimization_chunks = (32, 128, 256, 1)
    optimization_space = ()
    speculative_head = None

    @classmethod
    def compose(
        cls,
        checkpoint,
        *,
        rows,
        chunk,
        max_requests,
        max_context,
        provider="cuda",
        phase=None,
        **model_options,
    ):
        """Call the model module's compose function with this declaration's defaults.

        A family implements this one function in model.py. Subclasses declare
        their checkpoint reader and options; they do not invent another entry.
        """
        module = import_module(cls.__module__)
        composer = getattr(module, "compose", None)
        if not callable(composer):
            raise TypeError(
                f"{cls.__module__} must define compose(checkpoint, *, rows, chunk, max_requests, max_context, provider, phase)"
            )
        result = composer(
            checkpoint,
            rows=rows,
            chunk=chunk,
            max_requests=max_requests,
            max_context=max_context,
            provider=provider,
            phase=phase,
            **{**getattr(cls, "model_defaults", {}), **model_options},
        )
        if not isinstance(result, Composition):
            raise TypeError(f"{cls.__module__}.compose must return Composition")
        return result

    def __new__(
        cls,
        checkpoint,
        bundle,
        device,
        *,
        graphs=True,
        checkpoint_options=None,
        capacity=None,
        context=None,
        **model_options,
    ):
        from akas.artifacts.read import read_program

        for name, value in (("capacity", capacity), ("context", context)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")

        program = read_program(bundle)
        for requested, field in ((capacity, "max_requests"), (context, "max_context")):
            if requested is not None and requested != program[field]:
                raise ValueError(f"requested {field} differs from the compiled program")
        compiled = {
            **getattr(cls, "model_defaults", {}),
            **program.get("model_options", {}),
        }
        for name, requested in model_options.items():
            if name not in compiled or compiled[name] != requested:
                raise ValueError(
                    f"requested {name} differs from the compiled program; compile that option first"
                )
        loaded = cls.checkpoint_type(checkpoint, **(checkpoint_options or {}))
        return Executor.from_program(loaded, bundle, device, graphs=graphs)
