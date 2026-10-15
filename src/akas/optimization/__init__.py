"""Choose, qualify and publish an inference deployment from workload intent."""


def optimize(*args, **kwargs):
    from akas.optimization.supervisor import optimize as run

    return run(*args, **kwargs)


__all__ = ["optimize"]
