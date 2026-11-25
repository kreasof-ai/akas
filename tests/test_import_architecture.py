"""Enforce Python architecture rules without importing AKAS."""

import ast
from collections import defaultdict
from importlib.util import resolve_name
import json
from pathlib import Path

import pytest

from support.architecture import INTERFACE_LIMIT, SymbolIndex


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).parent / "fixtures/import-architecture.json"


def _imports(node, eager=True, runtime=True):
    match node:
        case ast.Import() | ast.ImportFrom():
            yield node, eager, runtime
            return
        case ast.FunctionDef() | ast.AsyncFunctionDef():
            eager = False
        case ast.If(
            test=ast.Name(id="TYPE_CHECKING")
            | ast.Attribute(value=ast.Name(id="typing"), attr="TYPE_CHECKING")
        ):
            for statement in node.body:
                yield from _imports(statement, eager, runtime=False)
            for statement in node.orelse:
                yield from _imports(statement, eager, runtime)
            return
    for child in ast.iter_child_nodes(node):
        yield from _imports(child, eager, runtime)


def _targets(node, package, modules):
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name, ""
        return
    base = resolve_name("." * node.level + (node.module or ""), package)
    for alias in node.names:
        child = f"{base}.{alias.name}"
        if child in modules:
            yield child, ""
        else:
            yield base, alias.name


def _private(name):
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _private_owner(target, member, path):
    parts = target.split(".")
    if _private(member):
        if path.stem == "__init__" or len(parts) == 2:
            return target
        return target.rpartition(".")[0]
    for index in range(len(parts) - 1, 0, -1):
        if _private(parts[index]):
            return ".".join(parts[:index])
    return None


def architecture_findings(root):
    src = root / "src"
    modules = {
        ".".join(path.relative_to(src).with_suffix("").parts).removesuffix(
            ".__init__"
        ): path
        for path in sorted((src / "akas").rglob("*.py"))
    }
    trees = {
        name: ast.parse(path.read_bytes(), filename=str(path))
        for name, path in modules.items()
    }
    packages = {
        name: name if path.stem == "__init__" else name.rpartition(".")[0]
        for name, path in modules.items()
    }
    paths = {name: path.relative_to(root).as_posix() for name, path in modules.items()}
    eager_imports, runtime_imports, private_imports = {}, {}, {}
    for source, tree in trees.items():
        for node, eager, runtime in _imports(tree):
            location = f"{paths[source]}:{node.lineno}"
            for target, member in _targets(node, packages[source], modules):
                if target not in modules or target == source:
                    continue
                if runtime:
                    runtime_imports.setdefault((source, target), location)
                    if eager:
                        eager_imports.setdefault((source, target), location)
                owner = _private_owner(target, member, modules[target])
                if owner and source != owner and not source.startswith(owner + "."):
                    imported = f"{target}.{member}" if member else target
                    private_imports.setdefault(f"{source} -> {imported}", location)
    symbols = SymbolIndex(trees, packages)
    references = tuple(symbols.references())
    tangled, tangles = _tangles(symbols, references, paths)
    return {
        "import_time_cycles": _cycle_findings(eager_imports),
        "module_cycles": _cycle_findings(runtime_imports),
        "private_imports": private_imports,
        "tangles": tangles,
        "field_leaks": _field_leaks(symbols, references, tangled, paths),
        "shallow_modules": _shallow_modules(symbols, references, paths),
    }


def _cycle_paths(edges):
    graph = defaultdict(list)
    for source, target in edges:
        graph[source].append(target)
    for source, target in edges:
        # An edge belongs to a cycle exactly when its target can reach its source.
        pending = [target]
        previous = {target: None}
        while pending and source not in previous:
            module = pending.pop()
            for dependency in graph[module]:
                if dependency not in previous:
                    previous[dependency] = module
                    pending.append(dependency)
        if source in previous:
            path = [source]
            while path[-1] != target:
                path.append(previous[path[-1]])
            yield (source, target), " -> ".join([source, *reversed(path)])


def _cycle_findings(imports):
    return {
        f"{source} -> {target}": f"{imports[source, target]}: {path}"
        for (source, target), path in _cycle_paths(imports)
    }


def _tangles(index, references, paths):
    edges = {}
    for ref in references:
        if ref.holder:
            source = index.symbols[ref.holder].owner
            target = index.symbols[ref.target].owner
            if source != target:
                edges.setdefault((source, target), f"{paths[ref.module]}:{ref.line}")
    cycles = dict(_cycle_paths(edges))
    neighbors = defaultdict(list)
    for source, target in cycles:
        neighbors[source].append(target)
    components = {}
    reportable = set()
    for source in neighbors:
        if source in components:
            continue
        pending, members = [source], {source}
        while pending:
            member = pending.pop()
            for target in neighbors[member]:
                if target not in members:
                    pending.append(target)
                    members.add(target)
        components.update(dict.fromkeys(members, source))
        modules = {index.symbols[member].module for member in members}
        if len(modules) > 1 or len(members) > INTERFACE_LIMIT:
            reportable.add(source)
    findings = {
        f"{source} -> {target}": f"{edges[source, target]}: {path}"
        for (source, target), path in cycles.items()
        if components[source] in reportable
    }
    return components, findings


def _field_leaks(index, references, tangled, paths):
    working, known_outside = set(), set()
    outside_operations = []
    for ref in references:
        if ref.target not in index.fields:
            continue
        owner = index.symbols[ref.target].owner
        user = index.symbols[ref.holder].owner if ref.holder else None
        if user == owner:
            if ref.holder in index.instance_methods and not ref.holder.endswith(
                ".__init__"
            ):
                working.add(owner)
        elif owner not in tangled or tangled.get(user) != tangled[owner]:
            known_outside.add(ref.target)
            if ref.operates_on_field:
                outside_operations.append(ref)
    guarded = working & {
        index.symbols[field].owner for field in index.fields - known_outside
    }
    findings = {}
    for ref in outside_operations:
        if index.symbols[ref.target].owner in guarded:
            key = f"{ref.holder or ref.module} -> {ref.target}"
            findings.setdefault(key, f"{paths[ref.module]}:{ref.line}")
    return findings


def _shallow_modules(index, references, paths):
    containers = {}
    for module in paths:
        parts = module.split(".")
        prefixes = (".".join(parts[:end]) for end in range(1, len(parts) + 1))
        containers[module] = tuple(prefix for prefix in prefixes if prefix in paths)
    totals = defaultdict(int)
    for function in index.functions:
        for scope in containers[index.symbols[function].module]:
            totals[scope] += 1
    exposed = defaultdict(dict)
    known_types = defaultdict(set)
    for ref in references:
        if ref.holder in index.functions and ref.target in index.classes:
            known_types[ref.holder].add(ref.target)
        if ref.target in index.functions:
            for scope in containers[index.symbols[ref.target].module]:
                if scope not in containers[ref.module]:
                    exposed[scope].setdefault(ref.target, ref)
    findings = {}
    for scope, functions in exposed.items():
        count, total = len(functions), totals[scope]
        if count <= INTERFACE_LIMIT or count * 2 < total:
            continue
        groups = defaultdict(int)
        for function in functions:
            types = {
                name
                for name in known_types[function]
                if scope in containers[index.symbols[name].module]
            }
            owner = index.symbols[function].owner
            for group in types if function == owner and types else (owner,):
                groups[group] += 1
        if len(groups) > INTERFACE_LIMIT or max(groups.values()) > INTERFACE_LIMIT:
            for function, ref in functions.items():
                findings[f"{scope} -> {function}"] = (
                    f"{paths[ref.module]}:{ref.line}: {scope} exposes {count} of {total} functions"
                )
    return findings


def _check(rule, findings, allowed=()):
    allowed = set(allowed)
    messages = [
        f"{rule}: {findings[key]} ({key})" for key in sorted(findings.keys() - allowed)
    ]
    messages.extend(
        f"{rule}: remove stale entry from {BASELINE.name}: {key}"
        for key in sorted(allowed - findings.keys())
    )
    if messages:
        pytest.fail("\n".join(messages), pytrace=False)


@pytest.fixture(scope="module")
def findings():
    return architecture_findings(ROOT)


def test_import_time_graph_is_acyclic(findings):
    _check("import-time cycle", findings["import_time_cycles"])


@pytest.mark.parametrize(
    ("rule", "key"),
    [
        ("module-cycle", "module_cycles"),
        ("private-import", "private_imports"),
        ("tangle", "tangles"),
        ("field-leak", "field_leaks"),
        ("shallow-module", "shallow_modules"),
    ],
    ids=["module-cycle", "private-import", "tangle", "field-leak", "shallow-module"],
)
def test_architecture_has_no_new_violations(findings, rule, key):
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    _check(rule, findings[key], baseline[key])


def _write_modules(tmp_path, modules):
    package = tmp_path / "src/akas"
    for filename, content in modules.items():
        path = package / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return architecture_findings(tmp_path)


def test_tangle_can_exist_without_an_import_cycle(tmp_path):
    result = _write_modules(
        tmp_path,
        {
            "queue/__init__.py": """\
from .timer import Timer

class Tasks:
    def next(self):
        return 1

    def advance(self, timer: Timer):
        return timer.tick(self)
""",
            "queue/timer.py": """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Tasks

class Timer:
    def tick(self, tasks: 'Tasks'):
        return tasks.next()
""",
        },
    )
    assert not result["module_cycles"]
    assert set(result["tangles"]) == {
        "akas.queue.Tasks -> akas.queue.timer.Timer",
        "akas.queue.timer.Timer -> akas.queue.Tasks",
    }


def test_tangle_follows_type_declarations(tmp_path):
    result = _write_modules(
        tmp_path,
        {
            "a.py": "from .b import B\nclass A:\n    peer: 'B'\n",
            "b.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .a import A\n"
            "class B:\n    def peer(self) -> 'A':\n        raise NotImplementedError\n",
        },
    )
    assert not result["module_cycles"]
    assert set(result["tangles"]) == {"akas.a.A -> akas.b.B", "akas.b.B -> akas.a.A"}


@pytest.mark.parametrize("size", [INTERFACE_LIMIT, INTERFACE_LIMIT + 1])
def test_tangle_flags_large_cycles_in_one_module(tmp_path, size):
    source = "\n".join(f"def f{i}(): return f{(i + 1) % size}()" for i in range(size))
    result = _write_modules(tmp_path, {"one.py": source})
    assert len(result["tangles"]) == (size if size > INTERFACE_LIMIT else 0)


def test_field_leak_resolves_receivers_and_direct_getters(tmp_path):
    result = _write_modules(
        tmp_path,
        {
            "queue.py": """\
class Queue:
    def __init__(self):
        self.items = []
        self._limit = 100

    def push(self, item):
        self.items.append(item)

    @property
    def pending(self):
        return self.items

    def contents(self):
        return self.items

class Record:
    def __init__(self):
        self.items = []
        self.label = 'plain data'

class InheritedQueue(Queue):
    pass

def make_queue() -> Queue:
    return Queue()
""",
            "client.py": """\
from typing import Optional
from .queue import Queue as WorkQueue, InheritedQueue, Record, make_queue

def through_annotation(queue: 'Optional[WorkQueue]'):
    queue.items.append(1)

def through_factory():
    queue = make_queue()
    return queue.pending[0]

def through_getter():
    WorkQueue().contents().clear()

def inherited_alias():
    items = InheritedQueue().contents()
    items.clear()

def replace(queue: WorkQueue):
    queue.items = []

def read(queue: WorkQueue):
    return queue.items

def record_access(record: Record):
    record.items.append(1)

def shadow(WorkQueue):
    return WorkQueue().items[0]
""",
        },
    )
    assert set(result["field_leaks"]) == {
        f"akas.client.{caller} -> akas.queue.Queue.items"
        for caller in (
            "through_annotation",
            "through_factory",
            "through_getter",
            "inherited_alias",
            "replace",
        )
    }


@pytest.mark.parametrize(
    ("used", "hidden", "shallow"), [(7, 0, False), (8, 8, True), (8, 9, False)]
)
def test_shallow_module_uses_width_and_information_hiding(
    tmp_path, used, hidden, shallow
):
    functions = [f"f{i}" for i in range(used)]
    definitions = "\n".join(
        f"def {name}(): return 1"
        for name in [*functions, *(f"internal{i}" for i in range(hidden))]
    )
    calls = "\n".join(f"{name}()" for name in functions)
    result = _write_modules(
        tmp_path,
        {
            "store.py": definitions,
            "facade.py": f"from .store import {', '.join(functions)}",
            "client.py": f"from .facade import {', '.join(functions)}\n{calls}",
        },
    )
    assert set(result["shallow_modules"]) == {
        f"akas.store -> akas.store.{name}" for name in functions if shallow
    }


def test_shallow_module_can_expose_one_wide_class(tmp_path):
    methods = [f"m{i}" for i in range(INTERFACE_LIMIT + 1)]
    definitions = "\n".join(f"    def {name}(self): return 1" for name in methods)
    calls = "\n".join(f"store.{name}()" for name in methods)
    result = _write_modules(
        tmp_path,
        {
            "store.py": f"class Store:\n{definitions}\n",
            "client.py": f"from . import store as api\nstore = api.Store()\n{calls}",
        },
    )
    assert set(result["shallow_modules"]) == {
        f"akas.store -> akas.store.Store.{name}" for name in methods
    }
