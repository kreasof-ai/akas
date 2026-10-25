from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/q4-word-prefill');root.mkdir(exist_ok=True)
s=Path('src/akas/ops/moe/cuda.py').read_text();s=s[s.index('def compact_experts('):s.index('\ndef packed_experts(')]
s=s.replace('def compact_experts(spec, schedule):','def factory(spec, schedule):')
s=s.replace('    routed = p.get("routed_input", False)','    from q4_word_tile import tile_reader\n    load_rhs=tile_reader(experts*columns,depth)\n    routed = p.get("routed_input", False)')
a=s.index('                    for i, j in T.Parallel(bn, bk):');b=s.index('                    T.gemm(lhs, rhs',a)
s=s[:a]+'                    load_rhs((w,),rhs,expert*columns+bx*bn,tile*bk)\n'+s[b:]
s=s.replace('threads=128) as (bx, group)','threads=p.get("threads",128)) as (bx, group)')
Path('build/lfm-moe/q4_word_prefill.py').write_text('from akas.kernels.kernel import Tensor,kernel,output\n'+s)
for bm,bn,bk,threads in ((64,64,64,128),(64,128,64,128),(64,128,64,256),(64,128,128,256)):
 for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
  p=dict(r=4096,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,tile_map=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},),block_m=bm,columns=bn,block_k=bk,threads=threads)
  path=root/f'{depth}-{bm}-{bn}-{bk}-{threads}.py'
  path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom q4_word_prefill import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
