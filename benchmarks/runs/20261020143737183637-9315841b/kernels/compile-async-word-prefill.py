from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/q4_word_prefill.py').read_text().replace('from q4_word_tile import tile_reader','from q4_async_word_tile import tile_reader').replace('num_stages=2','num_stages=p.get("stages",2)')
Path('build/lfm-moe/q4_async_word_prefill.py').write_text(s)
root=Path('build/lfm-moe/q4-async-word-prefill');root.mkdir(exist_ok=True)
for bm,bn,bk,threads,stages in ((64,128,64,128,2),(64,128,64,256,2),(64,128,64,128,3),(64,128,128,256,2)):
 for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
  p=dict(r=4096,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,tile_map=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},),block_m=bm,columns=bn,block_k=bk,threads=threads,stages=stages)
  path=root/f'{depth}-{bm}-{bn}-{bk}-{threads}-{stages}.py'
  path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom q4_async_word_prefill import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
