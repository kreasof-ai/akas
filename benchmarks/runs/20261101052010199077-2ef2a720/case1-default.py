from akas.convolution_tile_compat.factory import make
def tensor_export():
 return {"kernel":make({'r': 4, 'c': 2048, 'chunk': 2048, 'length': 3, 'columns': 8, 'threads': 256},False)}
