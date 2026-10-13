"""Independent numerical references for declared attention storage policies."""

def canonical_validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.quantization.kv.fp8 import decode_storage, encode_storage

    values = sample_arguments(signature)
    r, d = p["r"], p["d"]
    if kind == "rope":
        h = p["h"]
        values["positions"][:] = np.arange(r) * 17 + 1
        expected = {name: value.copy() for name, value in values.items()}
        x = values["x"].reshape(r,h,d).astype("float64")
        angle = values["positions"][:,None,None] * p["theta"]**(-2*np.arange(d//2)[None,None,:]/d)
        left, right = x[:,:,:d//2], x[:,:,d//2:]
        out = np.concatenate((left*np.cos(angle)-right*np.sin(angle),
                              left*np.sin(angle)+right*np.cos(angle)),axis=-1)
        expected["out"][:] = out.reshape(expected["out"].shape)
        return ValidationCase(values, expected, rtol=2e-4, atol=2e-5)

    h, kh, cap = p["h"], p["kh"], p["capacity"]
    chunk, splits = p.get("chunk",1), p.get("splits",16)
    values["enabled"][:] = 1
    if r > 1: values["enabled"][-1] = 0
    if "positions" in values:
        values["positions"][:] = np.minimum(cap-1, np.arange(r) % chunk + max(0,cap-chunk))
    quantized = p["kv_dtype"] not in ("float16","float32")
    rng = np.random.default_rng(29)
    for name in ("kc","vc"):
        if name not in values: continue
        if quantized:
            byte_shape = (*values[name].shape[:-1], d)
            bits = rng.integers(0,126,byte_shape,dtype="uint8")
            bits |= rng.integers(0,2,byte_shape,dtype="uint8") << 7
            values[name][:] = bits.view("uint32").reshape(values[name].shape) if p["provider"] == "webgpu" else bits
        elif p["kv_dtype"] == "float16":
            values[name][:] = values[name].astype("float16")
    for name in ("ks","vs"):
        if name in values: values[name][:] = np.abs(values[name]) + 0.01
    if kind == "attention_merge":
        parts = values["parts"].reshape(r,h,splits,d+2)
        parts[:,:,:,-1] = np.abs(parts[:,:,:,-1]) + 1
        expected = {name: value.copy() for name, value in values.items()}
        maxima = parts[:,:,:,-2]
        factors = np.exp(maxima-maxima.max(axis=2,keepdims=True))
        out = np.sum(parts[:,:,:,:d]*factors[:,:,:,None],axis=2) / np.sum(parts[:,:,:,-1]*factors,axis=2)[:,:,None]
        out[values["enabled"] == 0] = 0
        expected["out"][:] = out.reshape(expected["out"].shape)
        return ValidationCase(values, expected, rtol=2e-4, atol=2e-5)
    expected = {name: value.copy() for name, value in values.items()}
    if kind == "cache_write":
        for row in range(r):
            if not values["enabled"][row]: continue
            slot, position = row//chunk, values["positions"][row]
            for head in range(kh):
                for source, cache, scales in (("keys","kc","ks"),("values","vc","vs")):
                    channels = values[source][row,head*d:(head+1)*d]
                    if quantized:
                        blocks = (d+127)//128
                        encoded = np.empty(d,"uint8")
                        for block in range(blocks):
                            part = channels[block*128:min(d,(block+1)*128)]
                            scale = np.maximum(np.max(np.abs(part)),np.float32(1e-12)) / np.float32(448)
                            expected[scales][slot,head,position,block] = scale
                            encoded[block*128:min(d,(block+1)*128)] = encode_storage(part/scale)
                        expected[cache][slot,head,position] = encoded.view("uint32") if p["provider"] == "webgpu" else encoded
                    else:
                        expected[cache][slot,head,position] = channels.astype("float16") if p["kv_dtype"] == "float16" else channels
                        expected[scales][slot,head,position,0] = 1
        return ValidationCase(values, expected, rtol=0, atol=1e-9)

    def read(cache, scales, slot, head, end):
        data = values[cache][slot,head,:end]
        if quantized:
            bits = data.view("uint8").reshape(end,d) if p["provider"] == "webgpu" else data
            data = decode_storage(bits)
        factors = values[scales][slot,head,:end]
        return data.astype("float64") * np.repeat(factors,128,axis=-1)[:,:d]

    queries = values["q"].reshape(r,h,d).astype("float64")
    if kind == "attention_partial":
        out = expected["out"].reshape(r,h,splits,d+2)
    elif kind == "attention":
        out = expected["out"].reshape(r,h,d)
    else: raise ValueError(f"attention validation missing for {kind}")
    for row in range(r):
        length = min(cap,int(values["positions"][row])+1) if values["enabled"][row] else 0
        for head in range(h):
            keys = read("kc","ks",row//chunk,head//(h//kh),length)
            vals = read("vc","vs",row//chunk,head//(h//kh),length)
            if kind == "attention" and chunk > 1 and p.get("prefill_precision") == "float16":
                if not length:
                    out[row, head] = 0
                    continue
                query = queries[row, head].astype("float16").astype("float64")
                keys = keys.astype("float16").astype("float64")
                vals = vals.astype("float16").astype("float64")
                maximum, normalizer = -np.inf, 0.0
                accumulated = np.zeros(d, dtype="float64")
                for begin in range(0, length, 64):
                    scores = keys[begin:begin+64] @ query / np.sqrt(d)
                    next_maximum = max(maximum, scores.max())
                    correction = np.exp(maximum-next_maximum)
                    probabilities = np.exp(scores-next_maximum)
                    normalizer = normalizer*correction + probabilities.sum()
                    accumulated = accumulated*correction + probabilities.astype("float16").astype("float64") @ vals[begin:begin+64]
                    maximum = next_maximum
                out[row, head] = accumulated/normalizer
                continue
            scores = keys @ queries[row,head] / np.sqrt(d)
            if kind == "attention":
                if not length: out[row,head] = 0; continue
                probabilities = np.exp(scores-scores.max())
                out[row,head] = probabilities @ vals / probabilities.sum()
            else:
                span = (length+splits-1)//splits
                for part in range(splits):
                    begin,end = part*span,min(length,(part+1)*span)
                    if begin >= end:
                        out[row,head,part,:d] = 0
                        out[row,head,part,d:] = [-np.inf,0]
                        continue
                    maximum = scores[begin:end].max()
                    probabilities = np.exp(scores[begin:end]-maximum)
                    out[row,head,part,:d] = probabilities @ vals[begin:end]
                    out[row,head,part,d:] = [maximum,probabilities.sum()]
    return ValidationCase(values, expected, rtol=0.02, atol=0.002)



def mapped_validation(kind, p, signature, *, mapping=None):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))
    values = sample_arguments(signature)
    slots,chunk = p["slots"],p["chunk"]
    heads,kh,d,cap = p["heads"],p["kv_heads"],p["d"],p["capacity"]
    rows = slots*chunk
    values["mapping"][:] = np.arange(slots)[::-1] if mapping is None else mapping
    starts = np.minimum(np.arange(slots)*3,max(0,cap-chunk))
    if kind == "attention_qkv":
        values["positions"][:] = (starts[:,None]+np.arange(chunk)[None,:]).reshape(-1)
        values["active"][:] = 1
        if rows > 1: values["active"][-1] = 0
    else:
        values["positions"][:] = starts
        values["lengths"][:] = chunk
        if slots > 1: values["lengths"][-1] = max(1,chunk-1)
    expected = {name:value.copy() for name,value in values.items()}
    if kind == "attention_qkv":
        for row in range(rows):
            if not values["active"][row]: continue
            for head in range(heads+kh):
                query = head < heads
                offset = head*2*d if query else 2*heads*d+(head-heads)*d
                x = rounded(values["proj"][row,offset:offset+d]).astype("float64")
                w = values["qw" if query else "kw"]
                normalized = rounded(x/np.sqrt(np.mean(x*x)+p["eps"])*(1+w))
                angle = values["positions"][row]*p["theta"]**(-2*np.arange(32)/64)
                out = normalized.copy()
                left,right = normalized[:32],normalized[32:64]
                out[:32] = left*np.cos(angle)-right*np.sin(angle)
                out[32:64] = left*np.sin(angle)+right*np.cos(angle)
                out = rounded(out)
                if query:
                    expected["q"][row,head] = out
                else:
                    request,position = values["mapping"][row//chunk],values["positions"][row]
                    expected["kc"][request,head-heads,position] = out
                    start = 2*heads*d+kh*d+(head-heads)*d
                    expected["vc"][request,head-heads,position] = rounded(values["proj"][row,start:start+d])
    else:
        for batch in range(slots):
            request = values["mapping"][batch]
            for token in range(values["lengths"][batch]):
                row = batch*chunk+token
                length = min(cap,starts[batch]+token+1)
                for head in range(heads):
                    key_head = head//(heads//kh)
                    query = values["q"][row,head].astype("float64")
                    keys = values["kc"][request,key_head,:length]
                    vals = values["vc"][request,key_head,:length]
                    if keys.dtype == np.uint8:
                        from akas.quantization.weights.block_fp8 import decode_storage
                        keys = rounded(decode_storage(keys)*np.repeat(values["ks"][request,key_head,:length],128,axis=-1))
                        vals = rounded(decode_storage(vals)*np.repeat(values["vs"][request,key_head,:length],128,axis=-1))
                    keys, vals = keys.astype("float64"), vals.astype("float64")
                    scores = keys@query/np.sqrt(d)
                    # Reproduce declared BF16 probability staging independently.
                    maximum = -np.inf
                    normalizer,acc = 0.0,np.zeros(d,dtype="float64")
                    for start in range(0,length,64):
                        part = scores[start:start+64]
                        next_maximum = max(maximum,part.max())
                        correction = np.exp(maximum-next_maximum)
                        probability = np.exp(part-next_maximum)
                        normalizer = normalizer*correction+probability.sum()
                        acc = acc*correction+rounded(probability).astype("float64")@vals[start:start+64]
                        maximum = next_maximum
                    attended = rounded(acc/normalizer)
                    gate = rounded(values["proj"][row,head*2*d+d:head*2*d+2*d])
                    expected["out"][row,head*d:(head+1)*d] = rounded(attended/(1+np.exp(-gate)))
    return ValidationCase(values,expected)


def split_validation(kind, p, signature):
    """Split projection preparation and partitioned BF16/FP8 attention."""
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.quantization.kv.fp8 import encode_storage, decode_storage
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    values = sample_arguments(signature)
    rows = p.get("r", p.get("slots", 1) * p.get("chunk", 1))
    chunk = p.get("chunk", 1)
    if "active" in values:
        values["active"][:] = 1
        if rows > 1:
            values["active"][-1] = 0
    if "positions" in values:
        values["positions"][:] = np.minimum(np.arange(rows)*7, p["capacity"]-1)
    if "stats" in values:
        values["stats"][..., 1] = np.abs(values["stats"][..., 1]) + 1
    rng = np.random.default_rng(41)
    for name in ("kc", "vc"):
        if name in values and values[name].dtype == np.uint8:
            values[name][:] = rng.integers(32, 72, values[name].shape, dtype="uint8")
            values[name][:] |= rng.integers(0, 2, values[name].shape, dtype="uint8") << 7
    for name in ("ks", "vs"):
        if name in values:
            values[name][:] = 0.125
    expected = {name: value.copy() for name, value in values.items()}
    if kind in ("attention_q", "attention_kv"):
        d = values["w"].size
        query = kind == "attention_q"
        heads = values["q"].shape[1] if query else values["kc"].shape[1]
        for row in range(rows):
            for head in range(heads):
                if not values["active"][row]:
                    if query:
                        expected["q"][row, head] = 0
                    continue
                offset = head*d*(2 if query else 1)
                x = rounded(values["projection"][row, offset:offset+d]).astype("float64")
                normalized = rounded(x/np.sqrt(np.mean(x*x)+p["eps"])*(1+values["w"]))
                angle = values["positions"][row] * p["theta"]**(-2*np.arange(32)/64)
                out = normalized.copy()
                left, right = normalized[:32], normalized[32:64]
                out[:32] = left*np.cos(angle)-right*np.sin(angle)
                out[32:64] = left*np.sin(angle)+right*np.cos(angle)
                out = rounded(out)
                if query:
                    expected["q"][row, head] = out
                    continue
                slot, position = row//chunk, values["positions"][row]
                channels = rounded(values["values"][row, head*d:(head+1)*d])
                for source, cache, metadata in ((out, "kc", "ks"), (channels, "vc", "vs")):
                    if values[cache].dtype == np.uint8:
                        for block in range(d//128):
                            part = source[block*128:(block+1)*128]
                            scale = np.maximum(np.max(np.abs(part)), np.float32(1e-12)) / np.float32(448)
                            expected[metadata][slot, head, position, block] = scale
                            expected[cache][slot, head, position, block*128:(block+1)*128] = encode_storage(part/scale)
                    else:
                        expected[cache][slot, head, position] = source
    elif kind == "attention_partial":
        heads, d = values["q"].shape[1:]
        kh, splits = values["kc"].shape[1], p.get("splits", 16)
        for row in range(rows):
            length = int(values["positions"][row])+1 if values["active"][row] else 0
            tiles = ((length+63)//64 + splits-1)//splits
            for head in range(heads):
                cache_head = head//(heads//kh)
                for part in range(splits):
                    begin, end = part*tiles*64, min(length, (part+1)*tiles*64)
                    index = (row, cache_head, part, head % (heads//kh))
                    maximum, normalizer = -np.inf, 0.0
                    accumulated = np.zeros(d, dtype="float64")
                    for tile in range(begin, end, 64):
                        keys, vals = [values[name][row, cache_head, tile:min(tile+64,end)] for name in ("kc", "vc")]
                        if keys.dtype == np.uint8:
                            keys = decode_storage(keys)*np.repeat(values["ks"][row,cache_head,tile:min(tile+64,end)],128,axis=1)
                            vals = decode_storage(vals)*np.repeat(values["vs"][row,cache_head,tile:min(tile+64,end)],128,axis=1)
                        scores = rounded(keys).astype("float64") @ values["q"][row,head].astype("float64") / np.sqrt(d)
                        next_maximum = max(maximum, scores.max())
                        correction = np.exp(maximum-next_maximum)
                        probabilities = np.exp(scores-next_maximum)
                        normalizer = normalizer*correction+probabilities.sum()
                        accumulated = accumulated*correction+rounded(probabilities).astype("float64") @ rounded(vals).astype("float64")
                        maximum = next_maximum
                    expected["out"][index] = accumulated
                    expected["stats"][index] = [maximum, normalizer]
    elif kind == "attention_merge":
        d = values["partial"].shape[-1]
        for row in range(rows):
            for head in range(values["partial"].shape[1]*values["partial"].shape[3]):
                index = (row, head//8, slice(None), head%8)
                if not values["active"][row]:
                    expected["out"][row,head*d:(head+1)*d] = 0
                    continue
                stats = values["stats"][index]
                factors = np.exp(stats[:,0]-stats[:,0].max())
                out = np.sum(values["partial"][index]*factors[:,None],axis=0) / np.sum(stats[:,1]*factors)
                gate = rounded(values["projection"][row,head*2*d+d:head*2*d+2*d])
                expected["out"][row,head*d:(head+1)*d] = rounded(rounded(out)/(1+np.exp(-gate)))
    else:
        raise ValueError(f"missing split attention reference: {kind}")
    return ValidationCase(values, expected)


def validation(kind, parameters, signature):
    if "heads" in parameters and any(row["name"] == "mapping" for row in signature):
        return mapped_validation(kind, parameters, signature)
    if kind in ("attention_q", "attention_kv") or any(row["name"] == "stats" for row in signature):
        return split_validation(kind, parameters, signature)
    if kind == "attention" and "slots" in parameters and any(row["name"] == "projection" for row in signature):
        import numpy as np
        from akas.ops.contracts import ValidationCase

        specs = {row["name"]: row for row in signature}
        heads, d = specs["q"]["shape"][1:]
        slots = parameters["slots"]
        inputs = [{**row, "name": "proj" if row["name"] == "projection" else row["name"]} for row in signature]
        inputs.append(dict(name="mapping", shape=[slots], dtype="int32"))
        case = mapped_validation(kind, {**parameters, "heads":heads, "kv_heads":specs["kc"]["shape"][1], "d":d},
                                 inputs, mapping=np.arange(slots,dtype="int32"))
        def restore(values):
            return {"projection" if name == "proj" else name:value for name,value in values.items() if name != "mapping"}
        return ValidationCase(restore(case.arguments), restore(case.expected), case.rtol, case.atol)
    return canonical_validation(kind, parameters, signature)
