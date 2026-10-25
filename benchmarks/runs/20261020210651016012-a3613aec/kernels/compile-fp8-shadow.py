from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/fp8-shadow');root.mkdir(exist_ok=True)
for label,slots,chunk,heads,kh,cap in (('small',2,33,8,2,4096),('full',1,4096,32,8,131072)):
 p=dict(slots=slots,chunk=chunk,h=heads,kh=kh,d=64,capacity=cap)
 for op in ('extent','shadow'):
  path=root/f'{label}-{op}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom fp8_shadow import '+op+'_factory as factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n');build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
 for qm,threads,stages in ((64,128,1),(128,256,1),(128,256,2)) if label=='full' else ((128,256,1),):
  options=dict(r=slots*chunk,h=heads,kh=kh,d=64,capacity=cap,chunk=chunk,kv_dtype='float16',unit_scales=True,query_rows=qm,key_rows=64,threads=threads,stages=stages)
  path=root/f'{label}-attention-{qm}-{threads}-{stages}.py';path.write_text('from akas.ops.attention.cuda import packed_prefill\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":packed_prefill(KernelSpec("attention",'+repr(options)+'),{})}\n');build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
