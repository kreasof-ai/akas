"""Compare original numerical work independently of kernel authoring scaffolding."""

import ast
import copy
import hashlib


def _device(node):
    return any(
        (isinstance(d, ast.Attribute) and d.attr == "prim_func")
        or (
            isinstance(d.func if isinstance(d, ast.Call) else d, ast.Name)
            and (d.func if isinstance(d, ast.Call) else d).id == "kernel"
        )
        for d in node.decorator_list
    )


class Numerical(ast.NodeTransformer):
    def __init__(self, arguments):
        self.arguments = {name: f"argument_{i}" for i, name in enumerate(arguments)}

    def visit_Name(self, node):
        node.id = self.arguments.get(node.id, node.id)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if (
            node.attr == "packed_dtype"
            and isinstance(node.value, ast.Name)
            and node.value.id == "codec"
        ):
            return ast.Attribute(
                value=ast.Name(id="backend", ctx=ast.Load()),
                attr="PACKED_DTYPE",
                ctx=ast.Load(),
            )
        return node

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        # Codec tuples are compile-time ABI adapters, expanded before device IR.
        if node.func.id in ("read_weight", "decoder") and node.args:
            first = node.args[0]
            if isinstance(first, ast.Tuple) and len(first.elts) == 1:
                node.args[0] = first.elts[0]
        # The existing cache writer keeps its default F32 input policy. Its new
        # optional BF16 policy is qualified separately by codec execution tests.
        if node.func.id == "input_value" and len(node.args) == 3:
            return ast.Subscript(
                value=node.args[0],
                slice=ast.Tuple(elts=node.args[1:], ctx=ast.Load()),
                ctx=ast.Load(),
            )
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        # Frozen activation packing used the WebGPU provider. CUDA now selects
        # the same shared-memory branch without emitting subgroup intrinsics.
        if isinstance(node.op, ast.And):
            values = [
                v for v in node.values if ast.unparse(v) != "provider == 'webgpu'"
            ]
            if len(values) == 1:
                return values[0]
            node.values = values
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.left, ast.Name) or node.left.id != "kind":
            return node
        replacements = {
            "kind == 'ffn'": "paired",
            "kind == 'ffn_q16'": "paired",
            "kind in ('ffn', 'ffn_q16')": "paired",
            "kind == 'linear_add'": "add_residual",
            "kind == 'linear'": "not (paired or add_residual)",
            "kind == 'qnorm'": "query",
            "kind == 'kvnorm'": "not query",
        }
        text = replacements.get(ast.unparse(node))
        return ast.parse(text, mode="eval").body if text else node


def computation_hashes(source):
    """Freeze launch bodies and arithmetic inline helpers, not ABI adapters.

    Parameter names are normalized by their physical argument positions. Shapes,
    ABI order and result declarations have separate compiler/contract checks.
    Sets account for an unchanged body being shared or explicitly specialized.
    """
    from akas.kernels.kernel import lower

    hashes = set()
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, ast.FunctionDef):
            continue
        macro = any(
            isinstance(d, ast.Attribute) and d.attr == "macro"
            for d in function.decorator_list
        )
        if not macro and not _device(function):
            continue
        functional = any(
            isinstance(d.func if isinstance(d, ast.Call) else d, ast.Name)
            and (d.func if isinstance(d, ast.Call) else d).id == "kernel"
            for d in function.decorator_list
        )
        if functional:
            function, _ = lower(function)
        arguments = [argument.arg for argument in function.args.args]
        launches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "Kernel"
                for item in node.items
            )
        ]
        pieces = [("launch", launch) for launch in launches]
        if (
            macro
            and not launches
            and any(
                isinstance(node, (ast.BinOp, ast.UnaryOp))
                or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute))
                for statement in function.body
                for node in ast.walk(statement)
            )
        ):
            pieces.append(("inline", ast.Module(body=function.body, type_ignores=[])))
        for kind, piece in pieces:
            piece = Numerical(arguments).visit(copy.deepcopy(piece))
            hashes.add(
                kind
                + ":"
                + hashlib.sha256(
                    ast.dump(piece, include_attributes=False).encode()
                ).hexdigest()
            )
    return sorted(hashes)
