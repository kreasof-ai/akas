from akas.packed_expert_compat.factory import make
def tensor_export():
 return {"kernel":make('swiglu_experts',{'r': 4, 'top': 3, 'c': 513, 'expert_rows': True, 'input_dtype': 'bfloat16', 'output_dtype': 'bfloat16'},False,False)}
