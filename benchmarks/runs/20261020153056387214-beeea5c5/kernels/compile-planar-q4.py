"""Lossless Q4 byte-layout prototype: aligned quants followed by scale bits."""
from pathlib import Path
from akas.compilation.emit import build_artifact
source=Path('src/akas/quantization/weights/ggml/cuda.py').read_text()
s=source[source.index('def packed_dot_half('):source.index('\ndef supports(')]
s=s.replace('def packed_dot_half(shape):','def dot_half(shape):')
a=s.index('        base = row * row_bytes');b=s.index('        coefficients =',a)
s=s[:a]+'''        halves = T.meta_var(T.view(encoded[0], (words_count*2,), dtype="uint16"))
        scale = halves[shape[0]*depth//4 + row*(depth//32) + tile*8 + lane//4]
        packed = words[row*(depth//8) + tile*32 + lane]
'''+s[b:]
tile=Path('build/lfm-moe/q4_word_tile.py').read_text();tile=tile[tile.index('def tile_reader'):]
tile=tile.replace("words=T.view(encoded[0],(words_count,),dtype='uint32')","words=T.meta_var(T.view(encoded[0],(words_count,),dtype='uint32'))\n  halves=T.meta_var(T.view(encoded[0],(words_count*2,),dtype='uint16'))")
a=tile.index('    base=');b=tile.index('    lo0=',a)
tile=tile[:a]+'''    block=(row+i)*(depth//32)+channel//32
    scale=halves[rows*depth//4+block]
    packed=words[block*4+channel%16//4]
'''+tile[b:]
tile=tile.replace('T.import_source(CUDA_SOURCE)','T.import_source(_Q4_HALF2_SOURCE)')
Path('build/lfm-moe/planar_q4.py').write_text('from akas.quantization.weights.ggml.cuda import _Q4_HALF2_SOURCE\n'+s+'\n'+tile)
s=Path('src/akas/ops/moe/cuda.py').read_text();s=s[s.index('def packed_experts('):s.index('\ndef quantized_experts_warp(')]
s=s.replace('def packed_experts(spec: KernelSpec, schedule: Mapping[str, object]):','def factory(spec, schedule):').replace('    routed = p.get("routed_input", False)','    from planar_q4 import dot_half\n    dot = dot_half((experts*columns,depth))\n    routed = p.get("routed_input", False)')
Path('build/lfm-moe/planar_decode.py').write_text('from akas.kernels.kernel import Tensor,kernel,output\n'+s)
s=Path('build/lfm-moe/q4_word_prefill.py').read_text().replace('from q4_word_tile import tile_reader','from planar_q4 import tile_reader');Path('build/lfm-moe/planar_prefill.py').write_text(s)
root=Path('build/lfm-moe/planar-q4');root.mkdir(exist_ok=True)
for rows,experts,top,depth,columns,routed in ((1,64,4,2048,1536,False),(1,64,4,1536,2048,True),(1,4,2,2048,7,True),(4096,64,4,2048,1536,False),(4096,64,4,1536,2048,True)):
 p=dict(r=rows,experts=experts,top=top,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,weight_parts=({'rows':experts*columns,'format':'ggml','options':{'encoding':2}},))
 if rows>1:p.update(tile_map=True,block_m=64,columns=128,block_k=128,threads=256)
 module='planar_prefill' if rows>1 else 'planar_decode';path=root/f'{rows}-{depth}-{columns}.py'
 path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom '+module+' import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
