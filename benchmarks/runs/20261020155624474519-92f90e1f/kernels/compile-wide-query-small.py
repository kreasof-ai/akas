from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/wide-query-attention')
for qm in (128,256):
 p=dict(r=66,h=8,kh=2,d=64,capacity=4096,chunk=33,kv_dtype='fp8',provider='cuda',optimized=True,query_rows=qm,key_rows=64,stages=1,threads=512,group_queries=True,pipeline_group=8)
 path=root/f'{qm}-small.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom wide_query_attention import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("attention",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
