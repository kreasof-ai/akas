from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/fused-expert-activation')
for dtype in ('float32','float16'):
 p=dict(r=33,top=2,experts=4,k=256,o=65,weight_parts=({'rows':4*65,'format':'ggml','options':{'encoding':2}},),block_m=32,columns=64,block_k=64,threads=256,output_dtype=dtype)
 path=root/f'small-{dtype}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom fused_expert_activation import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
