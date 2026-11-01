from pathlib import Path
import json,re
from akas.benchmarking.history import Recorder
from akas.compilation.selection import source
from tensor.compiler.frontend.portable import export_spec
from tensor.compiler.cuda.lowering import lower_kernel

main=Path('/home/sagemaker-user/akas')
out=main/'build/lfm-serving/promoted-conversion-inspection'
out.mkdir(parents=True,exist_ok=True)
metadata=dict(runtime='akas',purpose='promoted-operand-lowering-inspection',provider='cuda',qualification_scope='Generated CUDA inspection only; no throughput or utilization claim')
rec=Recorder(main/'benchmarks',metadata,source_root=main,kind='diagnostic',change='Inspect existing FP8 to FP16 staging before reusing packed conversion',dependency_sources={'tensor':Path('/home/sagemaker-user/tensor')})
rec.artifact(Path(__file__),'inspect.py')
rows=[]
try:
 for name,kind,p,impl in [
  ('dense','projection_partial',dict(r=2048,k=2048,o=2048,partitions=1,block_m=16,columns=64,threads=128,stages=1,packed_copy=True,mma_dtype='float16'),'linear.pipeline:reference'),
  ('experts','expert_swiglu',dict(r=2048,k=2048,o=1536,top=4,experts=64,compact=True,tile_map=True,block_m=32,columns=64,threads=256,stages=1,activation_vector=16,mma_dtype='float16',weight_format='fp8',input_dtype='uint8'),'moe.cuda.fp8-paired-swiglu:reference')]:
  entry=out/(name+'.py'); entry.write_text(source(kind,p,contract='reference',target='sm_89',preferred=impl).render())
  spec=export_spec(entry,None)
  _,_,_,cuda=lower_kernel(spec['kernel'],'sm_89')
  path=out/(name+'.cu');path.write_text(cuda)
  row=dict(name=name,lines=len(cuda.splitlines()),casts=[line.strip() for line in cuda.splitlines() if any(term in line for term in ('to_half','cvt','fp8_e4m3','__half'))])
  rows.append(row); rec.artifact(entry,name+'.py'); rec.artifact(path,name+'.cu')
  print(name,json.dumps(row),flush=True)
 result=out/'results.json'; result.write_text(json.dumps(rows,indent=2)+'\n');rec.artifact(result,'results.json')
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed');raise
