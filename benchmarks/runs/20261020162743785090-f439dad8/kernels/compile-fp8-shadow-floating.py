from pathlib import Path
from akas.compilation.emit import build_artifact
source=Path('src/akas/ops/attention/cuda.py').read_text();a=source.index('def floating(spec:');b=source.index('\n\ndef ',a+5) if '\n\ndef ' in source[a+5:] else len(source);s=source[a:b]
s=s.replace('def floating(spec: KernelSpec, schedule: Mapping[str, object]):','def factory(spec, schedule):')
a=s.index('            tile_count = T.ceildiv(');b=s.index('            pipeline_tiles =',a)
s=s[:a]+'''            limits = T.alloc_fragment((bq,), "int32")
            live = T.alloc_fragment((1,), "int32")
            for i in T.Parallel(bq):
                row = slot*chunk+T.min(bx*bq+i,chunk-1)
                limits[i] = T.if_then_else(bx*bq+i<chunk and enabled[row]!=0, positions[row]+1, 0)
            T.reduce_max(limits,live,dim=0)
            tile_count = T.ceildiv(live[0],bk)
'''+s[b:]
Path('build/lfm-moe/shadow_floating.py').write_text('from akas.kernels.kernel import Tensor,kernel,output\n'+s)
root=Path('build/lfm-moe/fp8-shadow')
for label,slots,chunk,h,kh,cap in (('small',2,33,8,2,4096),('full',1,4096,32,8,131072)):
 for qm,threads,stages in (((16,128,2),(32,128,2),(64,128,2),(32,128,1)) if label=='full' else ((32,128,2),)):
  p=dict(r=slots*chunk,h=h,kh=kh,d=64,capacity=cap,chunk=chunk,kv_dtype='float16',unit_scales=True,query_rows=qm,key_rows=64,threads=threads,stages=stages,async_loads=True)
  path=root/f'{label}-floating-{qm}-{threads}-{stages}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom shadow_floating import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("attention",'+repr(p)+'),{})}\n');build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
