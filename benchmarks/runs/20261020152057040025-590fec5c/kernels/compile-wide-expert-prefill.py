from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/q4_paired_prefill.py').read_text().replace('bm not in (16, 32, 64, 128)','bm not in (16, 32, 64, 128, 256)')
Path('build/lfm-moe/q4_wide_prefill.py').write_text(s)
root=Path('build/lfm-moe/q4-wide-prefill');root.mkdir(exist_ok=True)
for bm,bn,bk,threads in ((128,128,64,512),(128,128,128,512),(256,64,64,512)):
 for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
  p=dict(r=4096,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,tile_map=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},),block_m=bm,columns=bn,block_k=bk,threads=threads)
  path=root/f'{depth}-{bm}-{bn}-{bk}-{threads}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom q4_wide_prefill import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
