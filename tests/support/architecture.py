"""Resolve explicit Python references for the architecture checks without imports."""

import ast
from collections import ChainMap, Counter
from importlib.util import resolve_name
from typing import NamedTuple


INTERFACE_LIMIT = 7
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCTIONS = DEFINITIONS[:2]
INNER_SCOPES = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class Reference(NamedTuple):
    module: str
    holder: str | None
    target: str
    line: int
    operates_on_field: bool = False


class Symbol(NamedTuple):
    module: str
    owner: str


def _scope_nodes(body, parent=None, grandparent=None):
    for node in body:
        yield node, parent, grandparent
        if not isinstance(node, DEFINITIONS + INNER_SCOPES):
            yield from _scope_nodes(ast.iter_child_nodes(node), node, parent)


def _import_bindings(node, package):
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.partition(".")[0]
            yield alias.asname or root, alias.name if alias.asname else root
    elif isinstance(node, ast.ImportFrom):
        base = resolve_name("." * node.level + (node.module or ""), package)
        for alias in node.names:
            if alias.name != "*":
                yield alias.asname or alias.name, f"{base}.{alias.name}"


def _arguments(node):
    args = node.args
    return [*args.posonlyargs, *args.args, *args.kwonlyargs]


def _annotation_expression(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            expression = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
        return ast.increment_lineno(expression, node.lineno - 1)
    return node


def _instance_parameter(node):
    if any(
        isinstance(decorator, ast.Name)
        and decorator.id in {"staticmethod", "classmethod"}
        for decorator in node.decorator_list
    ):
        return None
    args = node.args.posonlyargs or node.args.args
    return args[0].arg if args else None


def _field_operation(node, parent, grandparent):
    if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
        return True
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return True
    return (
        isinstance(parent, ast.Attribute)
        and parent.value is node
        and (
            not isinstance(parent.ctx, ast.Load)
            or isinstance(grandparent, ast.Call)
            and grandparent.func is parent
        )
    )


class SymbolIndex:
    def __init__(self, trees, packages):
        self.trees = trees
        self.packages = packages
        self.symbols = {}
        self.classes = {}
        self.functions = {}
        self.instance_methods = set()
        self.fields = set()
        self.field_annotations = {}
        self.field_types = {}
        self.return_types = {}
        self.module_names = {}
        self.aliases = {}
        self.getters = {}
        self.properties = set()
        self.bases = {}
        for module, tree in trees.items():
            names = self.module_names[module] = {}
            for node, _, _ in _scope_nodes(tree.body):
                if isinstance(node, DEFINITIONS):
                    key = names[node.name] = f"{module}.{node.name}"
                    self.symbols[key] = Symbol(module, key)
                    if isinstance(node, ast.ClassDef):
                        self.classes[key] = node
                        self._index_class(module, key, node)
                    else:
                        self.functions[key] = node
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    key = names[node.id] = f"{module}.{node.id}"
                    self.symbols[key] = Symbol(module, key)
                for local, original in _import_bindings(node, packages[module]):
                    names[local] = original
                    self.aliases[f"{module}.{local}"] = original
        for field, annotation in self.field_annotations.items():
            self.field_types[field] = self._annotation(
                annotation, self.module_names[self.symbols[field].module]
            )
        for key, node in self.functions.items():
            self.return_types[key] = self._annotation(
                node.returns, self.module_names[self.symbols[key].module]
            )
        for owner, node in self.classes.items():
            names = self.module_names[self.symbols[owner].module]
            self.bases[owner] = tuple(
                base
                for expression in node.bases
                if (base := self._annotation(expression, names)) is not None
            )

    def _index_class(self, module, owner, node):
        for member, _, _ in _scope_nodes(node.body):
            if isinstance(member, FUNCTIONS):
                key = f"{owner}.{member.name}"
                self.symbols[key] = Symbol(module, owner)
                self.functions[key] = member
                receiver = _instance_parameter(member)
                if receiver:
                    self.instance_methods.add(key)
                    for child, parent, _ in _scope_nodes(member.body):
                        if (
                            isinstance(child, ast.Attribute)
                            and isinstance(child.value, ast.Name)
                            and child.value.id == receiver
                            and isinstance(child.ctx, ast.Store)
                        ):
                            annotation = (
                                parent.annotation
                                if isinstance(parent, ast.AnnAssign)
                                else None
                            )
                            self._add_field(module, owner, child.attr, annotation)
                    body = member.body
                    if ast.get_docstring(member) is not None:
                        body = body[1:]
                    if len(body) == 1 and isinstance(body[0], ast.Return):
                        value = body[0].value
                        if (
                            isinstance(value, ast.Attribute)
                            and isinstance(value.value, ast.Name)
                            and value.value.id == receiver
                        ):
                            self.getters[key] = f"{owner}.{value.attr}"
                            if any(
                                isinstance(d, ast.Name) and d.id == "property"
                                for d in member.decorator_list
                            ):
                                self.properties.add(key)
            elif isinstance(member, ast.AnnAssign) and isinstance(
                member.target, ast.Name
            ):
                self._add_field(module, owner, member.target.id, member.annotation)

    def _add_field(self, module, owner, name, annotation):
        field = f"{owner}.{name}"
        self.fields.add(field)
        self.symbols.setdefault(field, Symbol(module, owner))
        if annotation is not None:
            self.field_annotations[field] = annotation

    def _canonical(self, name):
        visited = set()
        while name in self.aliases and name not in visited:
            visited.add(name)
            name = self.aliases[name]
        return name

    def _member(self, owner, member):
        target = self._canonical(f"{owner}.{member}")
        if target in self.symbols or owner not in self.bases:
            return target
        # Resolve inherited members when the indexed bases agree about ownership.
        pending = list(self.bases[owner])
        seen, candidates = {owner}, set()
        while pending:
            base = pending.pop()
            if base not in seen:
                seen.add(base)
                inherited = f"{base}.{member}"
                if inherited in self.symbols:
                    candidates.add(inherited)
                else:
                    pending.extend(self.bases.get(base, ()))
        return next(iter(candidates)) if len(candidates) == 1 else target

    def _resolve(self, node, names):
        match node:
            case ast.Name(id=name):
                return self._canonical(names.get(name))
            case ast.Attribute(value=value, attr=member):
                owner = self._resolve(value, names)
                if owner:
                    owner = self.field_types.get(owner) or owner
                    return self._member(owner, member)
            case ast.Call(func=func):
                constructor = self._resolve(func, names)
                if constructor in self.classes:
                    return constructor
                return self.return_types.get(constructor) or self.getters.get(
                    constructor
                )
        return None

    def _annotation(self, node, names):
        node = _annotation_expression(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if isinstance(node.right, ast.Constant) and node.right.value is None:
                node = node.left
            elif isinstance(node.left, ast.Constant) and node.left.value is None:
                node = node.right
        if isinstance(node, ast.Subscript):
            wrapper = self._resolve(node.value, names)
            if wrapper == "typing.Optional":
                return self._annotation(node.slice, names)
            if wrapper in {
                "typing.Annotated",
                "typing_extensions.Annotated",
            } and isinstance(node.slice, ast.Tuple):
                return self._annotation(node.slice.elts[0], names)
        target = self._resolve(node, names)
        return target if target in self.classes else None

    def _local_names(self, body, module, outer):
        names = outer.new_child()
        for node, _, _ in body:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names[node.id] = None
            if isinstance(node, DEFINITIONS):
                names[node.name] = None
            names.update(_import_bindings(node, self.packages[module]))
        return names

    def _infer_assignments(self, body, names):
        writes = Counter(
            node.id
            for node, _, _ in body
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        # Single assignments can carry constructor types and simple aliases.
        for node, _, _ in body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if writes[node.target.id] == 1:
                    names[node.target.id] = self._annotation(
                        node.annotation, names
                    ) or self._resolve(node.value, names)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and writes[target.id] == 1:
                        value = self._resolve(node.value, names)
                        if value in self.properties:
                            value = self.getters[value]
                        if value in self.symbols or value in self.trees:
                            names[target.id] = value

    def _scan_inner_scope(self, node, module, holder, outer):
        if isinstance(node, ast.Lambda):
            args = [*_arguments(node), node.args.vararg, node.args.kwarg]
            names = outer.new_child({arg.arg: None for arg in args if arg})
            yield from self._scan(_scope_nodes([node.body]), module, holder, names)
            return
        names = outer.new_child()
        for generator in node.generators:
            yield from self._scan(_scope_nodes([generator.iter]), module, holder, names)
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    names[target.id] = None
            yield from self._scan(_scope_nodes(generator.ifs), module, holder, names)
        values = (
            [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        )
        yield from self._scan(_scope_nodes(values), module, holder, names)

    def _scan(self, body, module, holder, names):
        for node, parent, grandparent in body:
            if isinstance(node, FUNCTIONS):
                yield from self._scan_function(node, module, holder, names)
                continue
            if isinstance(node, INNER_SCOPES):
                yield from self._scan_inner_scope(node, module, holder, names)
                continue
            if not isinstance(node, (ast.Name, ast.Attribute, ast.Call)):
                continue
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                continue
            target = self._resolve(node, names)
            if isinstance(node, ast.Call):
                callee = self._resolve(node.func, names)
                target = (
                    self.getters.get(callee) if callee not in self.properties else None
                )
                if callee in self.classes:
                    target = self._member(callee, "__init__")
            elif target in self.properties:
                field = self.getters[target]
                if field in self.fields:
                    yield Reference(module, holder, target, node.lineno)
                    target = field
            if target in self.symbols:
                yield Reference(
                    module,
                    holder,
                    target,
                    node.lineno,
                    target in self.fields
                    and _field_operation(node, parent, grandparent),
                )

    def _scan_function(self, node, module, enclosing, outer, symbol=None):
        body = list(_scope_nodes(node.body))
        names = self._local_names(body, module, outer)
        arguments = _arguments(node)
        for arg in arguments:
            names[arg.arg] = self._annotation(arg.annotation, outer)
        for arg in (node.args.vararg, node.args.kwarg):
            if arg:
                names[arg.arg] = None
                arguments.append(arg)
        annotations = [
            _annotation_expression(annotation)
            for annotation in [node.returns, *(arg.annotation for arg in arguments)]
        ]
        header = [
            *node.decorator_list,
            *node.args.defaults,
            *node.args.kw_defaults,
            *annotations,
        ]
        yield from self._scan(
            _scope_nodes(expression for expression in header if expression is not None),
            module,
            symbol or enclosing,
            outer,
        )
        if symbol and self.symbols[symbol].owner in self.classes:
            receiver = _instance_parameter(node)
            if receiver:
                names[receiver] = self.symbols[symbol].owner
        self._infer_assignments(body, names)
        yield from self._scan(body, module, symbol or enclosing, names)

    def references(self):
        for module, tree in self.trees.items():
            body = list(_scope_nodes(tree.body))
            names = ChainMap({}, self.module_names[module])
            self._infer_assignments(body, names)
            holders = {}
            for node, parent, grandparent in body:
                holder = holders.get(parent)
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    if len(targets) == 1 and isinstance(targets[0], ast.Name):
                        holder = f"{module}.{targets[0].id}"
                if holder:
                    holders[node] = holder
                if isinstance(node, ast.ClassDef):
                    owner = f"{module}.{node.name}"
                    yield from self._scan(
                        _scope_nodes(node.bases), module, owner, names
                    )
                    for method, _, _ in _scope_nodes(node.body):
                        if isinstance(method, FUNCTIONS):
                            key = f"{owner}.{method.name}"
                            yield from self._scan_function(
                                method, module, owner, names, key
                            )
                        elif isinstance(method, ast.AnnAssign):
                            annotation = _annotation_expression(method.annotation)
                            if annotation is not None:
                                yield from self._scan(
                                    _scope_nodes([annotation]), module, owner, names
                                )
                elif isinstance(node, FUNCTIONS):
                    key = f"{module}.{node.name}"
                    yield from self._scan_function(node, module, None, names, key)
                else:
                    yield from self._scan(
                        [(node, parent, grandparent)], module, holder, names
                    )
