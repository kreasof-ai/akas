from akas.kernels.definition import KernelSpec
from akas.ops.moe.cuda import tile_map
def tensor_export():
 return dict(kernel=tile_map(KernelSpec("expert_tiles",{'r': 33, 'rows': 33, 'block_m': 32, 'top': 8, 'experts': 256},"block-fp8"),{}))
