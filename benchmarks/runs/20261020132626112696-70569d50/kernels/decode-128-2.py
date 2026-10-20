from akas.ops.attention.cuda import mma_partial
from akas.kernels.definition import KernelSpec
def tensor_export():
 return {"kernel":mma_partial(KernelSpec("attention_partial",{'r': 1, 'h': 32, 'kh': 8, 'd': 64, 'capacity': 131072, 'kv_dtype': 'fp8', 'provider': 'cuda', 'splits': 64, 'key_rows': 128, 'stages': 2, 'optimized': True, 'decode_precision': 'float16'}),{})}
