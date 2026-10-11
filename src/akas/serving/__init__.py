"""Model-independent HTTP transport and caller-supplied service factories."""

from akas.serving.http import Service, create_app

__all__ = ["Service", "create_app"]
