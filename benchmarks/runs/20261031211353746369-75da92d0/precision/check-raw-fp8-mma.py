from pathlib import Path
import numpy as np
import tensor
from akas.compilation.selection import source
from akas.quantization._shared.e4m3 import encode_storage,decode_storage
root=Path('build/lfm-serving/raw-fp8-check');root.mkdir(parents=True,exist_ok=True)
p=dict(r=33,k=128,o=128,partitions=1,block_m=16,columns=64,threads=128,stages=1,packed_copy=True)
entry=root/'source.py';entry.write_text(source('projection_partial',p,contract='reference',target='sm_89').render())
tensor.build(entry,root/'kernel.tbin',target='sm_89')
rng=np.random.default_rng(41)
with tensor.Device() as device:
 kernel=device.load(root/'kernel.tbin')
 for mode in ('integer','fractional'):
  x=encode_storage(rng.choice([-192,-128,-32,0,32,128,192],(33,128)) if mode=='integer' else rng.normal(0,96,(33,128)))
  w=encode_storage(rng.choice([-192,-128,-32,0,32,128,192],(128,128)) if mode=='integer' else rng.normal(0,96,(128,128)))
  values=dict(x=x,w=w,activation_scales=np.ones((33,1),'float32'),scales=np.ones((1,1),'float32'),out=np.zeros((33,128),'float32'))
  args={d['name']:device.from_numpy(values[d['name']],dtype=d['dtype']) for d in kernel.manifest['arguments']}
  kernel.bind(*(args[d['name']] for d in kernel.manifest['arguments'])).launch()
  actual=args['out'].to_numpy();expected=(decode_storage(x).astype('float64')@decode_storage(w).astype('float64').T).astype('float32')
  print(mode,'different',np.count_nonzero(actual!=expected),'max',np.max(np.abs(actual-expected)),'relative',np.linalg.norm(actual-expected)/np.linalg.norm(expected),flush=True)
  np.savez(root/(mode+'.npz'),x=x,w=w,actual=actual,expected=expected)
  for value in args.values():value.release()
