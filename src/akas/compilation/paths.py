"""Locate installed dependency sources for producer provenance and tuning."""

from pathlib import Path


def tensor_source(relative=""):
    import tensor

    return Path(tensor.__file__).resolve().parent / relative


def source_path(value):
    path = Path(value)
    if path.is_absolute():
        try:
            relative = path.relative_to(Path(__file__).resolve().parents[3])
        except ValueError:
            return path
    else:
        relative = path
    parts = relative.parts
    if parts[:2] == ("src", "tensor"):
        return tensor_source(Path(*parts[2:]))
    return path
