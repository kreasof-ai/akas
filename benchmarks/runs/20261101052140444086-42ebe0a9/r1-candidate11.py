# DSL implementation SHA256: 0011cd405b8bedb43a6d849036fd32acbbb7068b45f7111b4f25edfc1163816b
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'conv', 'parameters': {'c': 2048, 'chunk': 2048, 'columns': 128, 'length': 3, 'r': 1, 'threads': 256, 'time_tile': 128}, 'phase': 'forward'}, {}, {'provider': 'cuda', 'target': 'sm_89'}, {'axes': ['columns', 'threads', 'time_tile'], 'contract': 'reference', 'defaults': {}, 'factory': 'parallel_prefill', 'identifier': 'convolution.cuda.parallel_prefill', 'kind': 'conv', 'module': 'akas.ops.convolution.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': ['sm_89']})}
