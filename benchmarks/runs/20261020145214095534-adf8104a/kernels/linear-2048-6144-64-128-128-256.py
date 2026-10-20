# DSL implementation SHA256: 260e4c79ce150df2bd7321f857ce82f71ca1473c92d3b74d587679594721c513
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'linear', 'parameters': {'block_k': 128, 'block_m': 64, 'block_n': 128, 'ggml': True, 'k': 2048, 'o': 6144, 'r': 4096, 'threads': 256, 'type': 2}, 'phase': 'forward'}, {}, {'provider': 'cuda', 'target': 'sm_89'}, {'axes': ['block_m', 'block_n', 'block_k', 'stages', 'threads'], 'contract': 'reference', 'defaults': {'block_k': 32, 'block_m': 32, 'block_n': 64, 'stages': 2}, 'factory': 'linear', 'identifier': 'linear.cuda.packed:prefill', 'kind': 'linear', 'module': 'akas.ops.linear.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml'], 'targets': []})}
