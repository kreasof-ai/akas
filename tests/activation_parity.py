"""Frozen pre-consolidation activation definitions and explicit ABI migration."""

import ast
import hashlib
import json
from pathlib import Path

BASELINE = json.loads((Path(__file__).parent/'fixtures/activation-baseline.json').read_text())


def source(name):
    row = BASELINE['sources'][name]
    assert hashlib.sha256(row['source'].encode()).hexdigest() == row['sha256']
    return row['source']


class WithoutExpertLayout(ast.NodeTransformer):
    """Remove only the added expert addressing; preserve original arithmetic."""
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id in ('expert_rows','top') for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_If(self,node):
        node = self.generic_visit(node)
        return node.orelse if ast.unparse(node.test) == 'expert_rows' else node

    def visit_IfExp(self,node):
        node = self.generic_visit(node)
        return node.orelse if ast.unparse(node.test) == 'expert_rows' else node


def old_factory(name):
    from types import ModuleType
    module = ModuleType('frozen_activation')
    tree = ast.parse(source(name))
    tree.body = [node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    module.swiglu_kernel = lambda p: p
    exec(compile(tree,'<frozen activation>','exec'),module.__dict__)
    return module.make_kernel


def old_dispatch(module,kind,parameters):
    name = {'packed_baseline':'cuda/packed_baseline.py','decode':'cuda/decode.py','bf16_batch':'cuda/bf16_batch.py'}[module.rsplit('.',1)[-1]]
    return old_factory(name)(kind,parameters)
