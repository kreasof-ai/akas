def make(p,before):
 from importlib import import_module
 from akas.kernels.definition import KernelSpec
 module=import_module("akas.convolution_tile_compat.before" if before else "akas.ops.convolution.cuda")
 return module.parallel_prefill(KernelSpec("conv",p,"reference"),{})
