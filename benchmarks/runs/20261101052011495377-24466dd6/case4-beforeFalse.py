from akas.paired_reduction_compat.factory import make
def tensor_export():
 return {"kernel":make('expert_partial',{'r': 33, 'rows': 33, 'k': 384, 'o': 256, 'experts': 7, 'top': 3, 'partitions': 3, 'block_m': 32, 'columns': 128, 'threads': 256, 'compact': False, 'tile_map': False, 'max_block_m': 128, 'stages': 2, 'activation_vector': 16},False,True)}
