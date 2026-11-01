def make(kind,p,before):
 from importlib import import_module
 from akas.kernels.definition import KernelSpec
 owner="residual" if kind=="add" else "normalization"
 module=import_module("akas.typed_activation_compat.before_"+owner if before else "akas.ops."+owner+".kernel")
 return getattr(module,kind)(KernelSpec(kind,p,"reference"),{})
