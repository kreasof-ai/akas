# DSL implementation SHA256: 3e6ce17f517108deca75e6bef3930beb3500d59c9b423a0706519b311d93e8fb
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'swiglu_experts', 'parameters': {'c': 1536, 'input_dtype': 'float32', 'output_dtype': 'bfloat16', 'partitions': 2, 'r': 1, 'round_dtype': 'bfloat16', 'round_inputs': True, 'top': 4}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {'expert_rows': True}, 'factory': 'swiglu_experts', 'identifier': 'akas.ops.activation.kernel:swiglu_experts', 'kind': 'swiglu_experts', 'module': 'akas.ops.activation.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
