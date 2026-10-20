from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/q4_paired_prefill.py').read_text().replace('from q4_word_tile import tile_reader','from register_weight_tile import reader as tile_reader')
s=s.replace('rhs = T.alloc_shared((bn, bk), "float16")','rhs = T.alloc_fragment((bn, bk), "float16")')
s=s.replace('T.gemm(lhs, rhs, accum, transpose_B=True)','T.gemm(lhs, rhs, accum, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)')
Path('build/lfm-moe/register_expert_prefill.py').write_text(s)
root=Path('build/lfm-moe/register-expert-prefill');root.mkdir(exist_ok=True)
items=[(33,4,2,256,128,False,32,64,64,128)]
items += [(4096,64,4,depth,columns,routed,bm,bn,bk,threads) for bm,bn,bk,threads in ((64,64,64,256),(128,64,64,256),(128,128,64,256)) for depth,columns,routed in ((2048,1536,False),(1536,2048,True))]
for rows,experts,top,depth,columns,routed,bm,bn,bk,threads in items:
 p=dict(r=rows,top=top,experts=experts,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,tile_map=True,weight_parts=({'rows':experts*columns,'format':'ggml','options':{'encoding':2}},),block_m=bm,columns=bn,block_k=bk,threads=threads)
 path=root/f'{rows}-{depth}-{columns}-{bm}-{bn}-{bk}-{threads}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom register_expert_prefill import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
 try:build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
 except Exception as error:print('REJECTED',path.name,str(error),flush=True)
