from pathlib import Path
import numpy as np
import torch
from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
root=Path('build/lfm-serving/raw-fp8-check')
for mode in ('integer','fractional'):
 d=np.load(root/(mode+'.npz'))
 a=torch.from_numpy(d['x']).cuda().view(torch.float8_e4m3fn);b=torch.from_numpy(d['w']).cuda().view(torch.float8_e4m3fn)
 result=w8a8_block_fp8_matmul_triton(a,b,torch.ones((33,1),device='cuda'),torch.ones((1,1),device='cuda'),[128,128],output_dtype=torch.float32).cpu().numpy()
 expected=d['expected'];native=d['actual']
 print(mode,'SG relative',np.linalg.norm(result-expected)/np.linalg.norm(expected),'max',np.max(np.abs(result-expected)),'native_equal',np.count_nonzero(native==result),flush=True)
 np.save(root/('sglang-'+mode+'.npy'),result)
