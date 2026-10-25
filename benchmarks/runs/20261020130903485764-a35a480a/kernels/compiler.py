from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/grouped-prefill');root.mkdir(exist_ok=True)
for dtype in ('float16','fp8'):
 for qm,bk in ((64,64),(128,64),(64,128)):
  p=dict(r=4096,h=32,kh=8,d=64,capacity=131072,chunk=4096,kv_dtype=dtype,unit_scales=dtype=='float16',query_rows=qm,key_rows=bk,stages=2,pipeline_group=8)
  path=root/f'{dtype}-{qm}-{bk}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom grouped_prefill import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n')
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',dtype,qm,bk,flush=True)
  p.update(r=33,h=8,kh=2,capacity=4096,chunk=33)
  small=root/f'{dtype}-{qm}-{bk}-small.py';small.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom grouped_prefill import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n')
  build_artifact(small,small.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT_SMALL',dtype,qm,bk,flush=True)
