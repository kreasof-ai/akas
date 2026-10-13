"""Model declarations open the same compiled program and request executor."""

from akas.execution.executor import Executor


class Definition:
    """Declare a checkpoint reader and a pure geometry-to-graph composition.

    Construction opens an already compiled program. Device allocation, phase
    selection and request state belong to Executor, never to a model subclass.
    """

    compilation_profile = "program"

    def __new__(cls, checkpoint, bundle, device, *, graphs=True,
                checkpoint_options=None, capacity=None, context=None, **model_options):
        from akas.artifacts.read import read_program

        program = read_program(bundle)
        for requested, field in ((capacity, "max_requests"), (context, "max_context")):
            if requested is not None and requested != program[field]:
                raise ValueError(f"requested {field} differs from the compiled program")
        compiled = {**getattr(cls, "model_defaults", {}), **program.get("model_options", {})}
        for name, requested in model_options.items():
            if name not in compiled or compiled[name] != requested:
                raise ValueError(f"requested {name} differs from the compiled program; compile that option first")
        loaded = cls.checkpoint_type(checkpoint, **(checkpoint_options or {}))
        return Executor.from_program(loaded, bundle, device, graphs=graphs)
