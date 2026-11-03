"""Operation-owned graph policies run before kernel selection and tuning."""


def resolve(policy, schedules, parameters, *, provider, target):
    """Select phase policies through the existing parameter-profile matcher.

    Entries named ``policy.<operation family>`` contain graph-policy settings.
    Their selectors use phase geometry; ordinary entries still select kernels.
    """
    options = {
        name: dict(settings) if isinstance(settings, dict) else settings
        for name, settings in (policy or {}).items()
    }
    if schedules is None:
        return options
    names = list(
        dict.fromkeys(
            entry["operation"][7:]
            for entry in getattr(schedules, "data", {}).get("entries", [])
            if entry["operation"].startswith("policy.")
        )
    )
    if not names:
        return options
    from akas.kernels.registry import planning_rules

    families = {
        module.__name__.split(".")[-2]
        for module in planning_rules()
        if hasattr(module, "lower")
    }
    if set(names) - families:
        raise ValueError(f"unknown operation policies: {sorted(set(names) - families)}")
    target = target or ("sm_89" if provider == "cuda" else "webgpu-portable-v1")
    for name in names:
        selected = schedules.select(
            "policy." + name, parameters, provider=provider, target=target
        )
        if selected:
            if not isinstance(options.get(name, {}), dict):
                raise ValueError(f"{name} policy must be a mapping")
            options[name] = {**options.get(name, {}), **selected}
    return options


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
    priority = {
        "moe": 0,
        "linear": 1,
        "attention": 2,
        "normalization": 3,
        "convolution": 4,
    }
    ordered = sorted(
        options.items(), key=lambda item: (priority.get(item[0], 2), item[0])
    )
    for operation, settings in ordered:
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
