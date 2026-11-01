def make(p,portable,before):
 from importlib import import_module
 name="kernel" if portable else "cuda"
 module=import_module("akas.map_compat.before_"+name if before else "akas.ops.moe."+name)
 from akas.kernels.definition import KernelSpec
 return module.expert_tile_map(KernelSpec("expert_tile_map",p,"reference"),{})
