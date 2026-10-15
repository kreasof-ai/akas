"""Device definitions expose their actual tensor results and preserve buffer ABIs."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from akas.kernels.kernel import lower


def definition(body):
    return ast.parse(textwrap.dedent(body)).body[0]


def test_results_lower_to_the_original_interleaved_buffer_order():
    function, signature = lower(
        definition("""
        def compute(x: Tensor((8,), "float32"), active: Tensor((1,), "int32")) -> Tensor((8,), "float32"):
            out = output((8,), "float32", position=1)
            out[0] = x[0] * active[0]
            return out
    """)
    )
    assert signature.inputs == ("x", "active")
    assert signature.outputs == ("out",)
    assert signature.mutates == ()
    assert signature.arguments == ("x", "out", "active")
    assert [arg.arg for arg in function.args.args] == list(signature.arguments)
    assert ast.unparse(function.body[0]) == "out[0] = x[0] * active[0]"
    assert len(function.body) == 1


def test_state_updates_and_multiple_results_are_explicit():
    _, signature = lower(
        definition("""
        def compute(x: Tensor((8,), "float32"), state: Tensor((8,), "float32")) -> tuple[Tensor((8,), "float32"), Tensor((8,), "float32")]:
            out = output((8,), "float32")
            state[0] = state[0] + x[0]
            out[0] = state[0]
            return state, out
    """)
    )
    assert signature.outputs == ("state", "out")
    assert signature.mutates == ("state",)
    assert signature.arguments == ("x", "state", "out")


@pytest.mark.parametrize(
    "source, message",
    [
        ('def compute(x) -> Tensor((8,), "float32"):\n    return x', "annotation"),
        ('def compute(x: Tensor((8,), "float32")):\n    return x', "annotate"),
        (
            'def compute(x: Tensor((8,), "float32")) -> Tensor((8,), "float32"):\n    x[0] = 0',
            "explicit result",
        ),
        (
            'def compute(x: Tensor((8,), "float32")) -> Tensor((8,), "float32"):\n    return missing',
            "declared outputs",
        ),
        (
            'def compute(x: Tensor((8,), "float32")) -> Tensor((8,), "float32"):\n    out = output((8,), "float32")\n    return x',
            "must be returned",
        ),
        (
            'def compute(x: Tensor((8,), "float32")) -> Tensor((8,), "float32"):\n    if True:\n        return x\n    return x',
            "unconditional",
        ),
        (
            'def compute(x: Tensor((8,), "float32")) -> Tensor((8,), "float32"):\n    a = output((8,), "float32", position=0)\n    b = output((8,), "float32", position=0)\n    return a, b',
            "positions",
        ),
    ],
)
def test_invalid_device_definitions_fail_locally(source, message):
    with pytest.raises(TypeError, match=message):
        lower(definition(source))


def test_return_types_must_match_declared_result_storage():
    with pytest.raises(TypeError, match="annotations must match"):
        lower(
            definition("""
            def compute(x: Tensor((8,), "float32")) -> Tensor((8,), "float16"):
                out = output((8,), "float32")
                out[0] = x[0]
                return out
        """)
        )


def test_every_device_definition_uses_the_functional_boundary():
    root = Path(__file__).parents[1] / "src/akas"
    count = 0
    for scope in (root / "ops", root / "quantization"):
        for path in scope.rglob("*.py"):
            if path == root / "ops/kernel.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.FunctionDef):
                    assert not any(
                        isinstance(d, ast.Attribute) and d.attr == "prim_func"
                        for d in node.decorator_list
                    ), path
                    for decorator in node.decorator_list:
                        target = (
                            decorator.func
                            if isinstance(decorator, ast.Call)
                            else decorator
                        )
                        if isinstance(target, ast.Name) and target.id == "kernel":
                            lower(node)
                            count += 1
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "primitive", path
    assert count > 100


def test_functional_declarations_import_without_a_compiler():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith('tensor.compiler') or name.split('.')[0] in {'tilelang', 'tvm', 'tvm_ffi'}:
            raise ImportError(name)
sys.meta_path.insert(0, Guard())
from akas.kernels.kernel import Tensor, kernel, output, lower
""",
        ],
        check=True,
    )


def test_specialization_preserves_ir_and_is_safe_across_threads():
    import tilelang
    import tilelang.language as T
    import tvm
    from akas.kernels.kernel import Tensor, kernel, output

    def make(count):
        @kernel
        def compute(x: Tensor((count,), "float32")) -> Tensor((count,), "float32"):
            out = output((count,), "float32")
            with T.Kernel(T.ceildiv(count, 32), threads=32) as block:
                for lane in T.Parallel(32):
                    index = block * 32 + lane
                    if index < count:
                        out[index] = x[index] * 2
            return out

        return compute

    def original(count):
        @T.prim_func
        def kernel(
            x: T.Tensor((count,), "float32"), out: T.Tensor((count,), "float32")
        ):
            with T.Kernel(T.ceildiv(count, 32), threads=32) as block:
                for lane in T.Parallel(32):
                    index = block * 32 + lane
                    if index < count:
                        out[index] = x[index] * 2

        return kernel

    with ThreadPoolExecutor(max_workers=3) as pool:
        kernels = list(pool.map(make, (32, 64, 96)))
    for count, observed in zip((32, 64, 96), kernels):
        assert tvm.ir.structural_equal(observed, original(count), map_free_vars=True)
        assert observed.__akas_signature__.arguments == ("x", "out")


def test_unreturned_input_writes_are_rejected_after_macro_expansion():
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel

    @T.macro
    def overwrite(x):
        x[0] = 0

    with pytest.raises(TypeError, match="writes must be returned"):

        @kernel
        def compute(
            x: Tensor((1,), "float32"), state: Tensor((1,), "float32")
        ) -> Tensor((1,), "float32"):
            with T.Kernel(1, threads=32):
                overwrite(state)
            return x


def test_attention_merge_takes_partials_as_input_and_returns_only_merged_output():
    from akas.ops.attention.cuda import flat

    result = flat(dict(h=4, d=64), 4)
    signature = result.__akas_signature__
    assert signature.inputs == ("parts",)
    assert signature.outputs == ("out",)
    assert signature.mutates == ()


def test_position_advance_declares_existing_positions_as_mutated_input():
    from akas.kernels.definition import KernelSpec
    from akas.ops.sequence.kernel import advance

    result = advance(KernelSpec("advance", dict(r=4)), {})
    assert result.__akas_signature__.inputs == ("positions", "enabled")
    assert result.__akas_signature__.outputs == ("positions",)
    assert result.__akas_signature__.mutates == ("positions",)


def test_quantized_results_include_all_metadata_in_the_format_owned_definition():
    from akas.kernels.definition import KernelSpec
    from akas.quantization.activations.q16.webgpu import quantize

    result = quantize(KernelSpec("quantize_q16", dict(r=4, k=64)), {})
    assert result.__akas_signature__.inputs == ("x",)
    assert result.__akas_signature__.outputs == ("packed", "scales", "sums")
    assert result.__akas_signature__.arguments == ("x", "packed", "scales", "sums")


def test_factory_dimensions_used_only_in_types_override_module_names(monkeypatch):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    monkeypatch.setitem(globals(), "shape", (99,))

    def make():
        shape = (16,)

        @kernel
        def compute(x: Tensor(shape, "float32")) -> Tensor(shape, "float32"):
            out = output(shape, "float32")
            with T.Kernel(1, threads=16):
                for lane in T.Parallel(16):
                    out[lane] = x[lane]
            return out

        return compute

    result = make()
    assert [
        tuple(int(d) for d in buffer.shape) for buffer in result.buffer_map.values()
    ] == [(16,), (16,)]


def test_unselected_branch_can_reference_an_unset_closure_cell():
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    def make(enabled=False):
        if enabled:
            optional = 3

        @kernel
        def compute(x: Tensor((1,), "float32")) -> Tensor((1,), "float32"):
            out = output((1,), "float32")
            with T.Kernel(1, threads=32):
                if enabled:
                    out[0] = x[0] * optional
                else:
                    out[0] = x[0]
            return out

        return compute

    assert make().__akas_signature__.arguments == ("x", "out")
    assert make(True).__akas_signature__.arguments == ("x", "out")
