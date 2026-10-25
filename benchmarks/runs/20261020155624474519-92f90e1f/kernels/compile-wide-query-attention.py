from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('src/akas/ops/attention/cuda.py').read_text();s=s[s.index('def packed_prefill('):s.index('\ndef ',s.index('def packed_prefill(')+5)]
s=s.replace('def packed_prefill(spec: KernelSpec, schedule: Mapping[str, object]):','def factory(spec,schedule):').replace('qm not in (64, 128)','qm not in (64, 128, 256)').replace('threads not in (128, 256)','threads not in (128, 256, 512)')
Path('build/lfm-moe/wide_query_attention.py').write_text('from akas.kernels.kernel import Tensor,kernel,output\n'+s)
root=Path('build/lfm-moe/wide-query-attention');root.mkdir(exist_ok=True)
for qm in (128,256):
 p=dict(r=4096,h=32,kh=8,d=64,capacity=131072,chunk=4096,kv_dtype='fp8',provider='cuda',optimized=True,query_rows=qm,key_rows=64,stages=1,threads=512,group_queries=True,pipeline_group=8)
 path=root/f'{qm}-64-512.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom wide_query_attention import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("attention",'+repr(p)+'),{})}\n')
 try:build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
 except Exception as error:print('REJECTED',path.name,str(error),flush=True)
