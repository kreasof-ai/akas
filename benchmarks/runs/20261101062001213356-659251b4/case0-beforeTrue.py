from akas.packed_expert_compat.factory import make
def tensor_export():
 return {"kernel":make('expert_partial',{'r': 1, 'rows': 1, 'k': 384, 'o': 256, 'experts': 7, 'top': 3, 'partitions': 1, 'block_m': 16, 'columns': 128, 'threads': 128, 'compact': True, 'tile_map': True, 'max_block_m': 128, 'stages': 1, 'activation_vector': 16, 'mma_dtype': 'float16', 'routed_input': True},True,False)}
