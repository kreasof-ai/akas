"""Compare the legacy valid-query domain before changing the arithmetic fixture."""
import ast,subprocess
from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/staged-floating-comparison');root.mkdir(exist_ok=True)
for label,text in [('original',subprocess.check_output(['git','show','HEAD:src/akas/ops/attention/cuda.py'],text=True)),('updated',Path('src/akas/ops/attention/cuda.py').read_text())]:
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='floating');function=ast.get_source_segment(text,node)
 module=root/f'{label}.py';module.write_text('from collections.abc import Mapping\nfrom akas.kernels.definition import KernelSpec\nfrom akas.kernels.kernel import Tensor,kernel,output\n'+function+'\ndef tensor_export():\n return {"kernel":floating(KernelSpec("attention",'+repr(dict(r=64,h=8,kh=2,d=64,capacity=512,chunk=32,kv_dtype='float16',unit_scales=True,prefill_precision='float16',optimized=True,async_loads=True,query_rows=32,key_rows=64,stages=1,threads=128))+'),{})}\n')
 build_artifact(module,module.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',label,flush=True)
