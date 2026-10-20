from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/fused-expert-activation');root.mkdir(exist_ok=True)
for dtype in ('float32','float16'):
 for bm,bn,bk,threads in ((64,64,64,256),(64,64,64,512),(32,64,64,256),(32,128,64,512)):
  p=dict(r=4096,top=4,experts=64,k=2048,o=1536,weight_parts=({'rows':64*1536,'format':'ggml','options':{'encoding':2}},),block_m=bm,columns=bn,block_k=bk,threads=threads,output_dtype=dtype)
  path=root/f'{dtype}-{bm}-{bn}-{bk}-{threads}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom fused_expert_activation import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n')
  try:build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
  except Exception as error:print('REJECTED',path.name,str(error),flush=True)
