"""Kernel definitions for kernels."""

import ast
import copy
import inspect
import linecache
import hashlib
import textwrap
from dataclasses import dataclass, replace
from threading import RLock


@dataclass(frozen=True)
class Signature:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    mutates: tuple[str, ...]
    arguments: tuple[str, ...]


def Tensor(shape, dtype):
    """A typed device tensor; accepts a flat count or a multidimensional shape."""
    import tilelang.language as T

    return T.Tensor(shape if isinstance(shape, (tuple, list)) else (shape,), dtype)


def output(shape, dtype, *, position=None):
    """Declare a returned buffer; position preserves an existing physical ABI.

    This declaration is interpreted by ``kernel`` and never runs on the device.
    Runtime allocation belongs to the enclosing execution plan.
    """
    raise TypeError("output declarations must be inside a @kernel function")


def lower(function):
    """Validate a functional definition and produce its explicit-buffer source.

    This part uses only Python's AST and can be checked without a compiler.
    """
    function = copy.deepcopy(function)
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise TypeError("device inputs must be explicit positional tensor arguments")
    if function.args.defaults or function.args.kw_defaults:
        raise TypeError("device inputs cannot have default values")
    if function.returns is None:
        raise TypeError("device definitions must annotate their returned tensors")
    arguments = [*function.args.posonlyargs, *function.args.args]
    if any((argument.annotation is None for argument in arguments)):
        raise TypeError("every device input must have a tensor or scalar annotation")
    if not function.body or not isinstance(function.body[-1], ast.Return):
        raise TypeError("a device definition must end with an explicit result return")
    returned = function.body[-1].value
    results = returned.elts if isinstance(returned, ast.Tuple) else [returned]
    if not results or any((not isinstance(value, ast.Name) for value in results)):
        raise TypeError("device results must be named buffers")
    names = tuple((value.id for value in results))
    if len(set(names)) != len(names):
        raise TypeError("device results must be unique")
    declarations, body = ([], [])
    for statement in function.body[:-1]:
        declaration = (
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and (statement.value.func.id == "output")
        )
        if not declaration:
            if any((isinstance(node, ast.Return) for node in ast.walk(statement))):
                raise TypeError(
                    "device definitions have one unconditional result return"
                )
            body.append(statement)
            continue
        if body and any(
            (
                not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
                for node in body
            )
        ):
            raise TypeError("declare result buffers before the computation")
        if len(statement.targets) != 1 or not isinstance(
            statement.targets[0], ast.Name
        ):
            raise TypeError("each result declaration needs one name")
        name = statement.targets[0].id
        call = statement.value
        if len(call.args) != 2 or any((key.arg != "position" for key in call.keywords)):
            raise TypeError("output expects shape, dtype and optional position")
        if name not in names or name in {arg.arg for arg in arguments}:
            raise TypeError("fresh outputs must be returned and cannot shadow inputs")
        position = None
        if call.keywords:
            position = ast.literal_eval(call.keywords[0].value)
            if type(position) is not int or position < 0:
                raise TypeError("output position must be a nonnegative integer")
        annotation = ast.Call(
            func=ast.Name(id="_akas_tensor", ctx=ast.Load()),
            args=call.args,
            keywords=[],
        )
        declarations.append((position, ast.arg(arg=name, annotation=annotation)))
    fresh = tuple((argument.arg for _, argument in declarations))
    if len(set(fresh)) != len(fresh):
        raise TypeError("fresh output names must be unique")
    inputs = tuple((argument.arg for argument in arguments))
    if set(names) - set(inputs) - set(fresh):
        raise TypeError("results must be input buffers or declared outputs")
    returned_types = (
        function.returns.slice.elts
        if isinstance(function.returns, ast.Subscript)
        and isinstance(function.returns.value, ast.Name)
        and (function.returns.value.id == "tuple")
        and isinstance(function.returns.slice, ast.Tuple)
        else [function.returns]
    )
    declared_types = {arg.arg: arg.annotation for arg in arguments}
    declared_types.update({arg.arg: arg.annotation for _, arg in declarations})

    def normalized_type(annotation):
        annotation = copy.deepcopy(annotation)
        if isinstance(annotation, ast.Call):
            if (
                isinstance(annotation.func, ast.Name)
                and annotation.func.id in ("Tensor", "_akas_tensor")
                or (
                    isinstance(annotation.func, ast.Attribute)
                    and annotation.func.attr == "Tensor"
                )
            ):
                annotation.func = ast.Name(id="Tensor", ctx=ast.Load())
                if (
                    annotation.args
                    and isinstance(annotation.args[0], (ast.Tuple, ast.List))
                    and (len(annotation.args[0].elts) == 1)
                ):
                    annotation.args[0] = annotation.args[0].elts[0]
        return ast.dump(annotation, include_attributes=False)

    total = len(arguments) + len(declarations)
    physical = [None] * total
    for position, argument in declarations:
        if position is not None:
            if position >= total or physical[position] is not None:
                raise TypeError("output positions must be unique and inside the ABI")
            physical[position] = argument
    if len(returned_types) != len(names) or any(
        (
            normalized_type(annotation) != normalized_type(declared_types[name])
            for name, annotation in zip(names, returned_types)
        )
    ):
        raise TypeError("returned tensor annotations must match their declared buffers")
    pending = iter(
        [*arguments, *(arg for position, arg in declarations if position is None)]
    )
    physical = [
        next(pending) if argument is None else argument for argument in physical
    ]
    function.args.posonlyargs = []
    function.args.args = physical
    function.body = body
    function.returns = None
    function.decorator_list = [ast.Name(id="_akas_prim_func", ctx=ast.Load())]
    ast.fix_missing_locations(function)
    signature = Signature(
        inputs,
        names,
        tuple((name for name in names if name in inputs)),
        tuple((arg.arg for arg in physical)),
    )
    return (function, signature)


def kernel(function=None, *, names=None):
    """Compile a typed inputs-to-results function without changing its body."""
    if function is None:
        caller_locals = dict(inspect.currentframe().f_back.f_locals)
        return lambda function: _compile(function, caller_locals, names)
    caller_locals = dict(inspect.currentframe().f_back.f_locals)
    return _compile(function, caller_locals, names)


def _compile(function, caller_locals, names):
    import tilelang.language as T

    source = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition, signature = lower(source.body[0])
    captured = {**function.__globals__, **caller_locals}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        try:
            captured[name] = cell.cell_contents
        except ValueError:
            continue
    names = dict(names or {})
    if set(names) - set(signature.arguments) or any(
        (
            not isinstance(value, str) or not value.isidentifier()
            for value in names.values()
        )
    ):
        raise TypeError(
            "kernel aliases must name existing arguments and valid identifiers"
        )
    physical_names = tuple((names.get(name, name) for name in signature.arguments))
    if len(set(physical_names)) != len(physical_names):
        raise TypeError("kernel aliases must be unique")
    if names:

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in names:
                    node.id = names[node.id]
                return node

            def visit_arg(self, node):
                node.arg = names.get(node.arg, node.arg)
                return node

        definition = Rename().visit(definition)
    captured["_akas_tensor"] = Tensor
    definition.decorator_list = []
    definition.name = "kernel"
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    text = ast.unparse(module) + "\n"
    filename = "<akas-kernel-" + hashlib.sha256(text.encode()).hexdigest() + ">"
    with _SOURCE_LOCK:
        linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
        try:
            exec(compile(ast.parse(text), filename, "exec"), captured)
            primitive = T.prim_func(captured["kernel"])
        finally:
            linecache.cache.pop(filename, None)
    from tvm import tirx

    parameters = {
        buffer.data: name
        for name, parameter in zip(physical_names, primitive.params)
        if (buffer := primitive.buffer_map.get(parameter)) is not None
    }
    writes = set()

    def visit(statement):
        if (
            isinstance(statement, tirx.BufferStore)
            and statement.buffer.data in parameters
        ):
            writes.add(parameters[statement.buffer.data])

    tirx.stmt_functor.post_order_visit(primitive.body, visit)
    undeclared = writes - {names.get(name, name) for name in signature.outputs}
    if undeclared:
        raise TypeError(
            f"device writes must be returned explicitly: {sorted(undeclared)}"
        )
    signature = replace(signature, arguments=physical_names)
    primitive.__akas_signature__ = signature
    return primitive


_SOURCE_LOCK = RLock()
