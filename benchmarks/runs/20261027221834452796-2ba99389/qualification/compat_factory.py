def before(spec,schedule):
 from akas.lfm3.before_linear import _packed
 return _packed({**spec.parameters,**schedule},paired=spec.kind=="ffn")
def after(spec,schedule):
 from akas.ops.linear.cuda import _packed
 return _packed({**spec.parameters,**schedule},paired=spec.kind=="ffn")
