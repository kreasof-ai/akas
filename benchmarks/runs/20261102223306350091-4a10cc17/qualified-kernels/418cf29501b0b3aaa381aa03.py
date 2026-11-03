# DSL implementation SHA256: d591e68a014148c3e920c31df4eca7c60f996c703a1be856b68e4ed14d2dc76d
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'quantize', 'parameters': {'input_dtype': 'float32', 'k': 11776, 'r': 1}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'quantize', 'identifier': 'akas.quantization.activations.fp8:reference', 'kind': 'quantize', 'module': 'akas.quantization.activations.fp8.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
