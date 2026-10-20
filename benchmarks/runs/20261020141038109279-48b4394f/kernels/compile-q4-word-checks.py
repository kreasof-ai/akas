from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/q4-word-prefill')
for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
 p=dict(r=4096,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,tile_map=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},),block_m=64,columns=64,block_k=64)
 path=root/f'baseline-{depth}.py';path.write_text('from akas.ops.moe.cuda import compact_experts\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":compact_experts(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BASELINE',depth,flush=True)
p=dict(r=33,top=2,experts=4,k=256,o=128,input_dtype='float32',routed_input=False,round_inputs=True,tile_map=True,weight_parts=({'rows':512,'format':'ggml','options':{'encoding':2}},),block_m=32,columns=64,block_k=64,threads=128)
path=root/'small.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom q4_word_prefill import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),{})}\n')
build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89')
path=root/'decode.py';path.write_text('''
import sys
sys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")
import tilelang.language as T
from akas.kernels.kernel import Tensor,kernel,output
from q4_word_tile import tile_reader
def tensor_export():
    load=tile_reader(128,2048)
    @kernel
    def compute(w:Tensor((128*2048//32*18,),"uint8"))->Tensor((128,2048),"float16"):
        out=output((128,2048),"float16")
        with T.Kernel(2,16,threads=128) as (row,column):
            values=T.alloc_shared((64,128),"float16")
            load((w,),values,row*64,column*128)
            T.copy(values,out[row*64,column*128])
        return out
    return {"kernel":compute}
''')
build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('CHECKS_COMPILED',flush=True)
