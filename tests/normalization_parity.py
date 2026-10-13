"""Frozen normalization sources and the precise dispatch-policy migration."""

import ast
import hashlib
import json
from pathlib import Path

BASELINE = json.loads((Path(__file__).parent / "fixtures/normalization-baseline.json").read_text())


def source(name):
    row = BASELINE["sources"][name]
    assert hashlib.sha256(row["source"].encode()).hexdigest() == row["sha256"]
    return row["source"]


def migrated_parameters(module, kind, parameters):
    if module.endswith(".reference"):
        return {"algorithm": "serial-tree", **parameters}
    if ".webgpu." in module and kind in ("rms", "add_rms"):
        return {"algorithm": "parallel-tree", **parameters}
    if kind in ("qnorm", "kvnorm"):
        return parameters
    tree = ast.parse(source(module.removeprefix("akas.ops.normalization.").replace(".", "/") + ".py"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {"rms_kernel": lambda p: p}
    exec(compile(tree, "<frozen normalization dispatch>", "exec"), namespace)
    result = namespace["make_kernel"](kind, parameters)
    # The unified definition derives fusion from the operation kind.
    result.pop("fused", None)
    return result
