from importlib import import_module
def before(spec,schedule):return import_module("akas.qualification.before_normalization")._serial({**spec.parameters,**schedule})
def after(spec,schedule):return import_module("akas.ops.normalization.kernel")._serial({**spec.parameters,**schedule})
