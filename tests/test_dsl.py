"""The package's producer entry files select actual lazy DSL factories."""

import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.parametrize("rows,chunk",[(1,1),(1,32),(4,1),(4,32)])
def test_common_packed_program_constructs_tirx(rows, chunk, tmp_path):
    pytest.importorskip("tilelang")
    import importlib
    import tvm
    from support.diagnostic import fixture
    from akas.models.lfm2.weights import LFM2Checkpoint
    from akas.models.lfm2._composition import program as compose_program
    from akas.compilation.compile import plan

    checkpoint = LFM2Checkpoint(fixture(tmp_path/"model.gguf"),tokenizer=False)
    graph = compose_program(checkpoint,rows=rows,chunk=chunk,max_requests=8,
                            max_context=128,provider="webgpu")
    compiled = plan(graph,provider="webgpu")
    for declaration in compiled.kernels:
        source = declaration.source
        kernel = getattr(importlib.import_module(source.module),source.factory)(*source.args)
        assert isinstance(kernel,tvm.tirx.PrimFunc)
        assert len(kernel.params) > 0


def test_factory_modules_import_without_frontend_or_core_compiler():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in {'tilelang','tvm','tvm_ffi','torch','triton'} or name.startswith('tensor.compiler'):
   raise ImportError(name)
sys.meta_path.insert(0, Guard())
from akas.kernels.definition import define, definition, KernelSpec, Capabilities
from support.lfm_profiles import CUDA_PROFILES
from akas.artifacts.schema import identity
assert len(identity('linear', {'k':256})) == 24
""",
            str(Path(__file__).resolve().parent),
        ],
        check=True,
    )
