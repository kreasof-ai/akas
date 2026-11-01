def make(kind,p,before):
 from importlib import import_module
 module=import_module("akas.precision_compat.before_moe" if before else "akas.ops.moe.cuda")
 return module._bucketed({**p,"rows":p["r"]},paired=kind!="expert_partial",quantized_output=kind=="fp8_expert_swiglu")
