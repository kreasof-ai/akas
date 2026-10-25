from pathlib import Path
import numpy as np
from factored_reference import inputs,expected
root=Path('build/lfm-moe/factored-scaled-attention');saved={}
arrays=inputs()
for length in (8192,131055):
 arrays['positions'][0]=length-1
 for bk in (64,128):
  saved[f'p{length}-bk{bk}']=expected(arrays,bk);print('REFERENCE',length,bk,flush=True)
np.savez_compressed(root/'expected.npz',**saved)
