from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/fp8-mma');root.mkdir(exist_ok=True)
text=Path('src/akas/ops/attention/cuda.py').read_text()
text=text[text.index('def mma_partial('):text.index('\ndef partial_flat(')]
text=text.replace('def mma_partial(', 'def factory(')
text=text.replace('p.get("kv_dtype") != "float16"', 'p.get("kv_dtype") != "fp8"').replace('        or not p.get("unit_scales")\n','')
text=text.replace('    @T.macro\n    def consume(', '''    from akas.quantization.kv.codec import get
    from akas.quantization._shared.access import packed_arguments,cache_arguments
    codec=get("fp8").bind((r,kh,cap,d),provider="cuda")
    decode=packed_arguments(codec.require("read_packed"))
    scalar=cache_arguments(codec.read)

    @T.macro
    def consume(''')
text=text.replace('kc: Tensor((r, kh, cap, d), "float16")','kc: Tensor((r, kh, cap, d), "uint8")').replace('vc: Tensor((r, kh, cap, d), "float16")','vc: Tensor((r, kh, cap, d), "uint8")')
text=text.replace('                    * ks[slot, head, 0, 0],', '                    ,')
text=text.replace('result[i, j] * vs[slot, head, 0, 0]', 'result[i, j]')
text=text.replace('            prob = T.alloc_shared((qm, bk), "float16")','''            encoded_k=T.alloc_shared((bk,d),codec.packed_dtype)
            encoded_v=T.alloc_shared((bk,d),codec.packed_dtype)
            scale_k=T.alloc_shared((bk,1),"float32")
            scale_v=T.alloc_shared((bk,1),"float32")
            keys=T.view(kc,dtype=codec.packed_dtype)
            values=T.view(vc,dtype=codec.packed_dtype)
            prob = T.alloc_shared((qm, bk), "float16")''')
text=text.replace('                    T.copy(kc[slot, head, start, 0], key)\n                    T.copy(vc[slot, head, start, 0], value)', '''                    T.copy(keys[slot,head,start,0],encoded_k)
                    T.copy(values[slot,head,start,0],encoded_v)
                    T.copy(ks[slot,head,start,0],scale_k)
                    T.copy(vs[slot,head,start,0],scale_v)
                    for i,j in T.Parallel(bk,d):
                        key[i,j]=decode(encoded_k,scale_k,i,j)
                        value[i,j]=decode(encoded_v,scale_v,i,j)''')
text=text.replace('kc[slot, head, T.min(start + i, cap - 1), j]', 'scalar(kc,ks,slot,head,T.min(start+i,cap-1),j)').replace('vc[slot, head, T.min(start + i, cap - 1), j]', 'scalar(vc,vs,slot,head,T.min(start+i,cap-1),j)')
Path('build/lfm-moe/fp8_mma.py').write_text('from akas.kernels.kernel import Tensor,kernel,output\nfrom akas.kernels.definition import KernelSpec\nfrom collections.abc import Mapping\n'+text)
for bk,stages in ((64,1),(64,2),(128,1),(128,2)):
 p=dict(r=1,h=32,kh=8,d=64,capacity=131072,kv_dtype='fp8',key_rows=bk,stages=stages,splits=16,optimized=True,decode_precision='float16')
 path=root/f'{bk}-{stages}.py'
 path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom fp8_mma import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n    return {"kernel":factory(KernelSpec("attention_partial",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89')
 print('BUILT',path,flush=True)
