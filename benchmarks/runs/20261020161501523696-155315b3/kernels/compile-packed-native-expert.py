from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/native_expert_prefill.py').read_text()
s=s.replace('    shape = (rows, top, depth) if routed else (rows, depth)','    shape = (rows * top, depth)')
s=s.replace('x: Tensor(shape, input_dtype)','x: Tensor(shape, "float16")')
s=s.replace('        counts: Tensor((experts,), "int32"),','        counts: Tensor((experts,), "int32"),\n        offsets: Tensor((experts+1,), "int32"),')
a=s.index('                    for i, j in T.Parallel(bm, bk):');b=s.index('                    T.copy(weights',a)
s=s[:a]+'                    T.copy(x[offsets[expert]+by*bm,tile*bk],lhs)\n'+s[b:]
Path('build/lfm-moe/packed_native_expert.py').write_text(s)
root=Path('build/lfm-moe/packed-native-expert');root.mkdir(exist_ok=True)
for bm,bn,bk,threads,stages in ((64,128,64,256,1),(64,128,64,256,2),(128,128,64,256,1),(128,128,64,256,2)):
 for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
  p=dict(r=4096,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,tile_map=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':1}},),block_m=bm,columns=bn,block_k=bk,threads=threads,stages=stages)
  path=root/f'{depth}-{bm}-{bn}-{bk}-{threads}-{stages}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom packed_native_expert import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
  try:build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
  except Exception as error:print('REJECTED',path.name,str(error),flush=True)
