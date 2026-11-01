from akas.reduction_compat.factory import make
def tensor_export():
 return {"kernel":make('fp8_expert_swiglu',{'r': 33, 'rows': 33, 'k': 384, 'o': 256, 'experts': 7, 'top': 3, 'partitions': 1, 'block_m': 32, 'columns': 128, 'threads': 256, 'compact': True, 'tile_map': True, 'max_block_m': 128, 'stages': 1, 'activation_vector': 16, 'product_dtype': 'bfloat16', 'activation_dtype': 'bfloat16'},False,False)}
