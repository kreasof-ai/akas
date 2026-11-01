from akas.precision_compat.factory import make
def tensor_export():
 return {"kernel":make('expert_swiglu',{'r': 33, 'rows': 33, 'k': 384, 'o': 256, 'experts': 7, 'top': 3, 'partitions': 1, 'block_m': 32, 'columns': 128, 'threads': 256, 'compact': True, 'tile_map': True, 'stages': 1, 'activation_vector': 16, 'mma_dtype': 'float16', 'pair_layout': 'wide', 'input_dtype': 'uint8', 'weight_format': 'fp8'},True)}
