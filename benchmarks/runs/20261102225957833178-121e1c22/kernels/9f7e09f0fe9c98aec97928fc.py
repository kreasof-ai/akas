# DSL implementation SHA256: 201502bf3d30a2b2b8c641ab397c54e33f2070f6af650f0c4bdcaefe36e03df2
from akas.compilation.transforms.state import make_kernel

def tensor_export():
    return {"kernel": make_kernel('akas.kernels.definition', 'define', [{'contract': 'reference', 'kind': 'attention', 'parameters': {'capacity': 2176, 'chunk': 2048, 'd': 64, 'h': 32, 'kh': 8, 'kv_dtype': 'float16', 'optimized': True, 'provider': 'cuda', 'r': 2048, 'unit_scales': True}, 'phase': 'prefill'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': ['async_loads', 'query_rows', 'key_rows', 'stages', 'threads', 'tma'], 'contract': 'reference', 'defaults': {'async_loads': True, 'key_rows': 64, 'prefill_precision': 'float16', 'query_rows': 32, 'stages': 2, 'threads': 128}, 'factory': 'floating', 'identifier': 'attention.cuda.pipelined', 'kind': 'attention', 'module': 'akas.ops.attention.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []}], [1, 2, 3, 4], 1, 256)}
