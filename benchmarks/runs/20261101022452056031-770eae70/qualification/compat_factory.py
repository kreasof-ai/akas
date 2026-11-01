def make(spec,schedule,before):
 from importlib import import_module
 owner="moe" if spec.kind=="expert_partial" else "linear"
 module=import_module("akas.qualification.before_"+owner if before else "akas.ops."+owner+".cuda")
 p={**spec.parameters,**schedule}
 return module._bucketed({**p,"rows":p["r"]}) if owner=="moe" else module._mma(p,grouped=False,prequantized=spec.kind=="projection_partial")
def before(spec,schedule):return make(spec,schedule,True)
def after(spec,schedule):return make(spec,schedule,False)
