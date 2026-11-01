from akas.reduction_compat.factory import make
def tensor_export():
 return {"kernel":make('expert_partial',{'r': 1, 'rows': 1, 'k': 512, 'o': 256, 'experts': 64, 'top': 4, 'partitions': 4, 'block_m': 16, 'columns': 64, 'threads': 128, 'compact': False, 'tile_map': False, 'max_block_m': 128, 'stages': 2, 'activation_vector': 16},False,True)}
