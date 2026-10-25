import sys
sys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")
from grouped_prefill import factory
def tensor_export():
 return {"kernel":factory({'r': 4096, 'h': 32, 'kh': 8, 'd': 64, 'capacity': 131072, 'chunk': 4096, 'kv_dtype': 'float16', 'unit_scales': True, 'query_rows': 64, 'key_rows': 128, 'stages': 2, 'pipeline_group': 8})}
