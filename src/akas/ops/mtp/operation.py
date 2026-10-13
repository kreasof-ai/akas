"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = tuple(item for contract in ("bf16", "block-fp8")
                        for item in variants(("mtp_join", "mtp_cast"), "mtp.cuda", contract))


def validation(kind, parameters, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    def rounded(values):
        return decode_bfloat16(encode_bfloat16(values))

    values = sample_arguments(signature)
    expected = {name:value.copy() for name,value in values.items()}
    if kind == "mtp_cast":
        expected["out"][:] = rounded(values["x"])
        return ValidationCase(values, expected, rtol=0, atol=0)
    width = parameters["c"]
    for index, name in enumerate(("embedding", "hidden")):
        source = values[name].astype("float64")
        out = source / np.sqrt(np.mean(source*source,axis=1,keepdims=True)+parameters["eps"])
        out *= 1+values[name+"_weight"]
        expected["out"][:,index*width:(index+1)*width] = rounded(out)
    return ValidationCase(values, expected)


VALIDATIONS = {name:validation for name in ("mtp_join", "mtp_cast")}


from akas.ops.contracts import OutputABI

OUTPUT_ARGUMENTS = {}

for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update({'mtp_join': OutputABI((-1,), ('out',)), 'mtp_cast': OutputABI((-1,), ('out',))})
