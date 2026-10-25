from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/factored_scaled_attention.py').read_text()
a=s.index('    @T.macro\n    def consume(');b=s.index('\n    @kernel',a);body=s[a:b]
body=body.replace('start + j < length,','start + j < length and i < group,')
body=body.replace('prob[i,j] = scores[i,j] * value_scale[j,0] * 128','prob[i,j] = T.if_then_else(i < group, scores[i,j] * value_scale[j,0] * 128, 0)')
s=s[:a]+body+s[b:]
Path('build/lfm-moe/masked_softmax_attention.py').write_text(s)
root=Path('build/lfm-moe/masked-softmax-attention');root.mkdir(exist_ok=True)
for bk in (64,128):
 p=dict(r=1,h=32,kh=8,d=64,capacity=131072,splits=64,chunk=1,kv_dtype='fp8',provider='cuda',optimized=True,key_rows=bk,stages=1)
 path=root/f'131072-{bk}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom masked_softmax_attention import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("attention_partial",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
