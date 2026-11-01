def make(kind,p,before,legacy=False):
 from importlib import import_module
 from akas.kernels.definition import KernelSpec
 owner="activation" if kind in ("swiglu","swiglu_experts") else "linear" if kind=="expert_merge" else "moe"
 module=import_module("akas.packed_expert_compat.before_"+owner if before else "akas.ops."+owner+(".kernel" if owner=="activation" else ".cuda"))
 if owner=="activation":return module._swiglu(p)
 if owner=="linear":return module.fp8(KernelSpec(kind,p,"reference"),{})
 if legacy:return module.partial(KernelSpec(kind,p,"reference"),{})
 return module._bucketed(p,paired=kind!="expert_partial",quantized_output=kind=="fp8_expert_swiglu")
