from akas.quantization.weights.ggml.cuda import _Q4_HALF2_SOURCE
def dot_half(shape):
    """A warp dots 256 values, rounding both operands to FP16 before F32 FMA.

    Keep the original GGUF bytes. A word view provides aligned loads and a
    funnel shift joins the alternating halfword-aligned Q4 blocks. No derived
    cache or fully dequantized matrix is allocated. ``offset`` selects one
    routed activation row from the caller's flat activation view. The final
    ``previous`` argument carries the lane accumulator across tiles, preserving
    continuous F32 FMA order instead of summing separately rounded tile dots.
    """
    import tilelang.language as T
    from math import prod

    depth = shape[-1]
    words_count = prod(shape) // 32 * 18 // 4
    row_bytes = depth // 32 * 18

    @T.macro
    def dot(encoded, x, offset, tile, lane, row, previous):
        T.import_source(_Q4_HALF2_SOURCE)
        words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
        halves = T.meta_var(T.view(encoded[0], (words_count*2,), dtype="uint16"))
        scale = halves[shape[0]*depth//4 + row*(depth//32) + tile*8 + lane//4]
        packed = words[row*(depth//8) + tile*32 + lane]
        coefficients = T.alloc_local((8,), "float32")
        for half in T.unroll(2):
            lo_bits = T.call_extern(
                "uint32",
                "akas_q4_adjacent_pair",
                packed >> (half * 16),
                T.cast(scale, "uint32"),
            )
            hi_bits = T.call_extern(
                "uint32",
                "akas_q4_adjacent_pair",
                packed >> (half * 16 + 4),
                T.cast(scale, "uint32"),
            )
            coefficients[half * 2] = T.cast(
                T.reinterpret("float16", T.cast(lo_bits, "uint16")), "float32"
            )
            coefficients[half * 2 + 1] = T.cast(
                T.reinterpret("float16", T.cast(lo_bits >> 16, "uint16")), "float32"
            )
            coefficients[4 + half * 2] = T.cast(
                T.reinterpret("float16", T.cast(hi_bits, "uint16")), "float32"
            )
            coefficients[5 + half * 2] = T.cast(
                T.reinterpret("float16", T.cast(hi_bits >> 16, "uint16")), "float32"
            )
        j = offset + tile * 256 + lane // 4 * 32 + lane % 4 * 4
        lo = T.alloc_local((4,), "float32")
        hi = T.alloc_local((4,), "float32")
        for component in T.vectorized(4):
            lo[component] = T.cast(T.cast(x[j + component], "float16"), "float32")
            hi[component] = T.cast(T.cast(x[j + 16 + component], "float16"), "float32")
        result = T.alloc_var("float32")
        result = previous
        for component in T.unroll(4):
            result += lo[component] * coefficients[component]
        for component in T.unroll(4):
            result += hi[component] * coefficients[component + 4]
        return result

    return dot


def tile_reader(rows,depth):
 import tilelang.language as T
 row_bytes=depth//32*18;words_count=rows*row_bytes//4
 @T.macro
 def load(encoded,destination,row,column):
  T.import_source(_Q4_HALF2_SOURCE)
  words=T.meta_var(T.view(encoded[0],(words_count,),dtype='uint32'))
  halves=T.meta_var(T.view(encoded[0],(words_count*2,),dtype='uint16'))
  for i,j in T.Parallel(destination.shape[0],destination.shape[1]//8):
   local=j//4*32+j%4*4;channel=column+local
   if row+i<rows and channel+19<depth:
    block=(row+i)*(depth//32)+channel//32
    scale=halves[rows*depth//4+block]
    packed=words[block*4+channel%16//4]
    lo0=T.call_extern('uint32','akas_q4_adjacent_pair',packed,T.cast(scale,'uint32'))
    lo1=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>16,T.cast(scale,'uint32'))
    hi0=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>4,T.cast(scale,'uint32'))
    hi1=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>20,T.cast(scale,'uint32'))
    destination[i,local]=T.reinterpret('float16',T.cast(lo0,'uint16'))
    destination[i,local+1]=T.reinterpret('float16',T.cast(lo0>>16,'uint16'))
    destination[i,local+2]=T.reinterpret('float16',T.cast(lo1,'uint16'))
    destination[i,local+3]=T.reinterpret('float16',T.cast(lo1>>16,'uint16'))
    destination[i,local+16]=T.reinterpret('float16',T.cast(hi0,'uint16'))
    destination[i,local+17]=T.reinterpret('float16',T.cast(hi0>>16,'uint16'))
    destination[i,local+18]=T.reinterpret('float16',T.cast(hi1,'uint16'))
    destination[i,local+19]=T.reinterpret('float16',T.cast(hi1>>16,'uint16'))
   else:
    for offset in T.unroll(4):
     destination[i,local+offset]=0;destination[i,local+offset+16]=0
 return load
