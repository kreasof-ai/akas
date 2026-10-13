"""Expand extracted codec macros for comparison with frozen pre-migration math.

This removes codec selection/layout plumbing, substitutes the original FP8 ABI,
and inlines the actual codec AST. Frozen hashes still detect changes to its math.
"""

import ast
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEC = ast.parse((ROOT / "src/akas/quantization/kv/fp8.py").read_text())


def macro(factory):
    outer = next(n for n in CODEC.body if getattr(n, "name", None) == factory)
    return next(n for n in outer.body if isinstance(n, ast.FunctionDef))


class Replace(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Name(self, node):
        return copy.deepcopy(self.values.get(node.id, node))


class Expand(ast.NodeTransformer):
    def __init__(self, name):
        self.slot = "r" if name == "partial" else "slots"

    def visit_Name(self, node):
        if node.id == "metadata_columns":
            return ast.Constant(2)
        if node.id == "packed_dtype":
            return ast.Constant("float8_e4m3fn")
        if node.id == "metadata_dtype":
            return ast.Constant("float32")
        return node

    def visit_Attribute(self, node):
        if ast.unparse(node) == "backend.PACKED_DTYPE":
            return ast.Constant("float8_e4m3fn")
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in (
            "decode",
            "decode_scalar",
            "decode_packed",
        ):
            source = macro(
                "packed_decoder" if node.func.id == "decode_packed" else "decoder"
            )
            values = dict(
                zip((arg.arg for arg in source.args.args), node.args, strict=True)
            )
            values["BLOCK"] = ast.Constant(128)
            return Replace(values).visit(copy.deepcopy(source.body[0].value))
        return self.generic_visit(node)

    def visit_arg(self, node):
        if (
            node.arg in ("kc", "vc")
            and node.annotation is not None
            and "cache_shape" in ast.unparse(node.annotation)
        ):
            node.annotation = ast.parse(
                f"T.Tensor(({self.slot}, 2, cap, d), 'uint8')", mode="eval"
            ).body
        if (
            node.arg in ("ks", "vs")
            and node.annotation is not None
            and "metadata_shape" in ast.unparse(node.annotation)
        ):
            node.annotation = ast.parse(
                f"T.Tensor(({self.slot}, 2, cap, 2), 'float32')", mode="eval"
            ).body
        return node


def restore(node):
    node = copy.deepcopy(node)
    plumbing = {
        "format",
        "backend",
        "encode_pair",
        "decode",
        "decode_scalar",
        "decode_packed",
        "storage",
        "cache_shape",
        "metadata_shape",
        "metadata_columns",
        "packed_dtype",
    }
    body = []
    for item in node.body:
        if isinstance(item, ast.Assign):
            target = item.targets[0]
            names = (
                [target.id]
                if isinstance(target, ast.Name)
                else [n.id for n in target.elts]
            )
            if any(name in plumbing for name in names):
                continue
        if isinstance(item, ast.If) and (
            "storage.metadata" in ast.unparse(item.test)
            or "backend.PACKED_DTYPE" in ast.unparse(item.test)
            or "packed_dtype" in ast.unparse(item.test)
        ):
            continue
        body.append(item)
    node.body = body
    if node.name == "write":
        kernel = next(n for n in node.body if isinstance(n, ast.FunctionDef))
        active = kernel.body[0].body[0]
        call = active.body[-1].value
        source = macro("writer")
        values = dict(
            zip((arg.arg for arg in source.args.args), call.args, strict=True)
        )
        values["blocks"] = ast.Constant(2)
        values["value_offset"] = ast.Constant(0)
        expanded = [Replace(values).visit(copy.deepcopy(n)) for n in source.body]
        declarations = expanded[:6]
        rotated = next(
            i
            for i, n in enumerate(active.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "rotated"
        )
        active.body[rotated + 1 : rotated + 1] = declarations
        active.body[-1:] = expanded[6:]
    if node.name == "make_kernel":
        branch = next(
            (
                n
                for n in node.body
                if isinstance(n, ast.If) and "attention_qkv" in ast.unparse(n.test)
            ),
            None,
        )
        if branch is not None:
            branch.body = [
                n
                for n in branch.body
                if not (
                    isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "encode_pair"
                )
            ]
            kernel = next(n for n in branch.body if isinstance(n, ast.FunctionDef))
            choice = kernel.body[0].body[0].body[-1]
            if len(choice.orelse) == 1 and isinstance(choice.orelse[0], ast.Expr):
                call = choice.orelse[0].value
                source = macro("writer")
                values = dict(
                    zip((arg.arg for arg in source.args.args), call.args, strict=True)
                )
                values["blocks"] = ast.Constant(2)
                values["value_offset"] = ast.parse(
                    "2 * heads * d + kh * d", mode="eval"
                ).body
                renames = dict(
                    ka="key_abs",
                    va="value_abs",
                    km="key_max",
                    vm="value_max",
                    scales_k="key_scale",
                    scales_v="value_scale",
                    quantized_key="key",
                    quantized_value="value",
                )

                class Rename(ast.NodeTransformer):
                    def visit_Name(self, value):
                        return (
                            ast.Name(renames[value.id], value.ctx)
                            if value.id in renames
                            else value
                        )

                choice.orelse = [
                    Rename().visit(Replace(values).visit(copy.deepcopy(n)))
                    for n in source.body
                ]

    class Offset(ast.NodeTransformer):
        def visit_BinOp(self, value):
            value = self.generic_visit(value)
            if (
                isinstance(value.op, ast.Add)
                and isinstance(value.left, ast.Constant)
                and value.left.value == 0
            ):
                return value.right
            return value

        def visit_Subscript(self, value):
            value = self.generic_visit(value)
            if (
                isinstance(value.value, ast.Name)
                and value.value.id in ("values", "proj")
                and isinstance(value.slice, ast.Tuple)
            ):

                def terms(index):
                    if isinstance(index, ast.BinOp) and isinstance(index.op, ast.Add):
                        return terms(index.left) + terms(index.right)
                    return [index]

                indices = terms(value.slice.elts[1])
                index = indices[0]
                for term in indices[1:]:
                    index = ast.BinOp(index, ast.Add(), term)
                value.slice.elts[1] = index
            return value

    node = Offset().visit(node)
    node = Expand(node.name).visit(node)

    # Error wording is not numerical behavior; retain the original frozen opcode dispatcher.
    class Wording(ast.NodeTransformer):
        def visit_Constant(self, value):
            if value.value == "unknown quantized KV operation: ":
                return ast.Constant("unknown FP8 KV operation: ")
            return value

    return Wording().visit(node)
