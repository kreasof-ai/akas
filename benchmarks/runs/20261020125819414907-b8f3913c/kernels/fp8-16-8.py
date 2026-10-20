import sys
sys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")
from group_fp8_8 import factory
from akas.kernels.definition import KernelSpec
def tensor_export():
 return {"kernel":factory(KernelSpec("attention_partial",{'r': 1, 'h': 32, 'kh': 8, 'd': 64, 'capacity': 131072, 'kv_dtype': 'fp8', 'key_rows': 64, 'stages': 2, 'splits': 16, 'optimized': True, 'unit_scales': False, 'decode_precision': 'float16'}),{})}
