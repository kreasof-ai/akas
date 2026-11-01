from akas.reduction_compat.factory import make
def tensor_export():
 return {"kernel":make('expert_swiglu',{'r': 4, 'rows': 4, 'k': 384, 'o': 256, 'experts': 7, 'top': 3, 'partitions': 1, 'block_m': 16, 'columns': 32, 'threads': 128, 'compact': True, 'tile_map': True, 'max_block_m': 128, 'stages': 1, 'activation_vector': 16, 'mma_dtype': 'float16', 'input_dtype': 'uint8', 'weight_format': 'fp8', 'product_dtype': 'bfloat16', 'activation_dtype': 'bfloat16', 'output_dtype': 'bfloat16'},False,False)}
