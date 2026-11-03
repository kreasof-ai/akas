# DSL implementation SHA256: d591e68a014148c3e920c31df4eca7c60f996c703a1be856b68e4ed14d2dc76d
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'quantize', 'parameters': {'input_dtype': 'float32', 'k': 2048, 'r': 1}, 'phase': 'decode'}, {'groups_per_block': 4}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': ['groups_per_block'], 'contract': 'reference', 'defaults': {}, 'factory': 'warp_quantize', 'identifier': 'fp8.cuda.warp_quantize', 'kind': 'quantize', 'module': 'akas.quantization.activations.fp8.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': ['r', 'k', 'top', 'input_dtype', 'routed_input'], 'targets': []})}
