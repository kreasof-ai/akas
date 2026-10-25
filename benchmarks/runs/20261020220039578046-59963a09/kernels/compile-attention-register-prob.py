from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/attention-register-prob');root.mkdir(exist_ok=True)
for name,slots,chunk,h,kh,cap in (('small',2,33,8,2,4096),('full',1,4096,32,8,131072)):
    p=dict(r=slots*chunk,h=h,kh=kh,d=64,capacity=cap,chunk=chunk,kv_dtype='float16',unit_scales=True,query_rows=64,key_rows=64,threads=128,stages=1,async_loads=True)
    source=root/(name+'.py')
    source.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom attention_register_prob import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel": factory(KernelSpec("attention",'+repr(p)+'),{})}\n')
    build_artifact(source,source.with_suffix('.tbin'),provider='cuda',target='sm_89')
    print('BUILT',name,flush=True)
