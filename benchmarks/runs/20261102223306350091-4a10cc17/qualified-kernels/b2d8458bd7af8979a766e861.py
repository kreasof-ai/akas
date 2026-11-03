# DSL implementation SHA256: bce473231bb00e2dd7df45b3cb8016a6da9e6f819b214d333493b83ad1669d6b
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'attention_merge', 'parameters': {'capacity': 2176, 'd': 64, 'h': 32, 'kh': 8, 'kv_dtype': 'float16', 'optimized': True, 'provider': 'cuda', 'r': 1, 'splits': 16, 'unit_scales': True}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {'prefill_precision': 'float16'}, 'factory': 'merge_floating', 'identifier': 'attention.cuda.floating', 'kind': 'attention_merge', 'module': 'akas.ops.attention.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
