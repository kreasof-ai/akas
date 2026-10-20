import sys
sys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")
from prefill_occupancy import factory
def tensor_export():
 return {"kernel":factory({'r': 4096, 'h': 32, 'kh': 8, 'd': 64, 'capacity': 131072, 'chunk': 4096, 'kv_dtype': 'float16', 'unit_scales': True, 'query_rows': 64, 'key_rows': 64, 'stages': 1, 'pipeline_group': 8, 'threads': 128, 'group_queries': True})}
