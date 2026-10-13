"""Operation-owned graph policies run before kernel selection and tuning."""


def lower(graph, commit=None, *, policy=None, provider="cuda", target=None):
    """Apply explicit pipeline choices without model or runtime dispatch.

    Each operation owns its policy keys and numerical lowerings. New models
    automatically use them when their graph contains the relevant operations.
    Kernel qualification and beam search run on the resulting physical calls.
    """
    if not policy:
        return graph, commit
    from akas.ops.registry import planning_rules

    options = dict(policy or {})
    callbacks = {
        module.__name__.split(".")[-2]: module.lower
        for module in planning_rules()
        if hasattr(module, "lower")
    }
    unknown = options.keys() - callbacks.keys()
    if unknown:
        raise ValueError(f"unknown operation policies: {sorted(unknown)}")
    for operation, settings in options.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{operation} policy must be a mapping")
        graph, commit = callbacks[operation](
            graph, commit, settings, provider=provider, target=target
        )
    return graph, commit
