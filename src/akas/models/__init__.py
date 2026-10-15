"""Model families own their public exports and composition declarations."""


def resolve_export(name):
    """Compatibility for historical root imports; prefer a family import."""
    from importlib import import_module
    from pkgutil import iter_modules

    if name.startswith("_"):
        raise AttributeError(name)
    matches = []
    for entry in iter_modules(__path__):
        if entry.ispkg and not entry.name.startswith("_"):
            family = import_module(f"{__name__}.{entry.name}")
            if name in getattr(family, "__all__", ()):
                matches.append(getattr(family, name))
    if len(matches) != 1:
        raise AttributeError(name)
    return matches[0]
