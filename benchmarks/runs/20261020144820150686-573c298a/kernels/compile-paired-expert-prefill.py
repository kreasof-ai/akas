from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/q4_word_prefill.py').read_text()
s=s.replace('bm not in (16, 32, 64)','bm not in (16, 32, 64, 128)')
s=s.replace('groups = (rows * top + bm - 1) // bm + experts','map_block = min(bm, 64)\n    pairs = bm // map_block\n    groups = (rows * top + map_block - 1) // map_block + experts')
s=s.replace('layout = bm.bit_length() - 3','layout = map_block.bit_length() - 3')
s=s.replace('            by = tile_rows[layout, group]','            mapped_row = tile_rows[layout, group]\n            by = mapped_row // pairs')
s=s.replace('if expert >= 0 and by * bm < counts[expert]:','if expert >= 0 and mapped_row % pairs == 0 and by * bm < counts[expert]:')
Path('build/lfm-moe/q4_paired_prefill.py').write_text(s)
root=Path('build/lfm-moe/q4-paired-prefill');root.mkdir(exist_ok=True)
for bm,bn,bk,threads in ((128,64,64,256),(128,128,64,256),(128,64,128,256)):
 for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
  p=dict(r=4096,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,tile_map=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},),block_m=bm,columns=bn,block_k=bk,threads=threads)
  path=root/f'{depth}-{bm}-{bn}-{bk}-{threads}.py'
  path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom q4_paired_prefill import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
