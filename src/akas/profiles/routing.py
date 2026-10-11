"""Lazy producer routing shared by execution profiles."""

from importlib import import_module


def operation(module, kind, parameters):
    return import_module("akas.ops." + module).make_kernel(kind, parameters)
