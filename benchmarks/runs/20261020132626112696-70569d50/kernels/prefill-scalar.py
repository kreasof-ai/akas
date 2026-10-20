from akas.ops.attention.cuda import floating
from akas.kernels.definition import KernelSpec
def tensor_export():
 return {"kernel":floating(KernelSpec("attention",{'r': 4096, 'h': 32, 'kh': 8, 'd': 64, 'capacity': 131072, 'chunk': 4096, 'kv_dtype': 'fp8', 'provider': 'cuda', 'unit_scales': False, 'key_rows': 64, 'query_rows': 32, 'stages': 0, 'async_loads': False}),{})}
