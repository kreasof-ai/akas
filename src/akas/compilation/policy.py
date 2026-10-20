"""Operation-owned graph policies run before kernel selection and tuning."""


def lower(graph, commit=None, *, policy=None, provider="cuda", target=None):
    """Apply explicit pipeline choices without model or runtime dispatch.

    Each operation owns its policy keys and numerical lowerings. New models
    automatically use them when their graph contains the relevant operations.
    Kernel qualification and beam search run on the resulting physical calls.
    """
    if not policy:
        return graph, commit
    from akas.kernels.registry import planning_rules

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


def candidates(graph, *, provider, target, commit=None):
    """Ask operations for applicable pipeline alternatives, including a baseline."""
    from akas.kernels.registry import planning_rules

    alternatives = [{}]
    combined = {}
    for module in planning_rules():
        propose = getattr(module, "search_policies", None)
        if propose is None:
            continue
        name = module.__name__.split(".")[-2]
        choices = propose(graph, commit, provider=provider, target=target)
        for choice in choices:
            alternatives.append({name: choice})
        if choices:
            combined[name] = choices[0]
    if len(combined) > 1:
        alternatives.append(combined)
    return alternatives


def combine(policies):
    """Keep individual choices and combine defaults across operation families.

    Decode and prefill can expose different policies. Extend each discovered
    choice with the first choice from other families, without merging settings
    within a family or enumerating a Cartesian product. These are candidates;
    compilation and model validation still decide which combinations are valid.
    """
    alternatives, defaults = [], {}
    for policy in policies:
        if policy not in alternatives:
            alternatives.append(policy)
        for name, settings in policy.items():
            defaults.setdefault(name, settings)
    for policy in tuple(alternatives):
        combined = {**defaults, **policy}
        if combined not in alternatives:
            alternatives.append(combined)
    return alternatives
