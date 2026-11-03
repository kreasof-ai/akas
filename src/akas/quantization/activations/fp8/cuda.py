"""CUDA specializations and shared bindings for fp8."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def quantize(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, top = (p["r"], p["k"], p.get("top"))
    rows = r * (top or 1)
    shape = (r, top, k) if top else (r, k)
    input_dtype = p.get("input_dtype", "bfloat16")
    storage = FORMAT.storage(shape, provider="cuda")
    scales_shape = storage.named["scales"].shape

    @kernel
    def compute(
        x: Tensor(shape, input_dtype),
    ) -> tuple[Tensor(shape, "uint8"), Tensor(scales_shape, "float32")]:
        out = output(shape, "uint8")
        scales = output(scales_shape, "float32")
        with T.Kernel(rows, k // 128, threads=128) as (b, tile):
            values = T.alloc_fragment((128,), "float32")
            absolute = T.alloc_fragment((128,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(128):
                if top:
                    values[j] = T.cast(x[b // top, b % top, tile * 128 + j], "float32")
                else:
                    values[j] = T.cast(x[b, tile * 128 + j], "float32")
                absolute[j] = T.abs(values[j])
            T.reduce_max(absolute, maximum, dim=0)
            for j in T.Parallel(128):
                scale = T.max(maximum[0], T.float32(1e-12)) / T.float32(448)
                bits = T.cast(
                    T.call_extern("uint32", "tensor_encode_e4m3", values[j] / scale),
                    "uint8",
                )
                if top:
                    out[b // top, b % top, tile * 128 + j] = bits
                else:
                    out[b, tile * 128 + j] = bits
            for j in T.Parallel(1):
                scale = T.max(maximum[0], T.float32(1e-12)) / T.float32(448)
                if top:
                    scales[b // top, b % top, tile] = scale
                else:
                    scales[b, tile] = scale
        return (out, scales)

    return compute


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="cuda")


def bind(shape):
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_tile
    from .kernel import reader

    storage = FORMAT.storage(shape, provider="cuda")
    read = reader(shape, provider="cuda")
    return Codec(storage, read, matrix_tile(read, shape))


AVAILABLE = "cuda" in dict(FORMAT.providers)
CAPABILITIES = ("dequantize", "read", "read_tile", "quantize")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "cuda")


def warp_quantize(spec: KernelSpec, schedule: Mapping[str, object]):
    """Assign one 128-value quantization block to each warp."""
    import tilelang.language as T
    from tensor.compiler.frontend.view import buffer_view
    p = {**spec.parameters, **schedule}
    r, k, top = p['r'], p['k'], p.get('top')
    rows = r * (top or 1)
    shape = (r,top,k) if top else (r,k)
    scales_shape = (*shape[:-1], k//128)
    groups = p.get('groups_per_block',4)
    if groups not in (2,4,8) or k%128:
        raise ValueError('invalid warp quantization geometry')
    dtype = p.get('input_dtype','bfloat16')
    @kernel
    def compute(x:Tensor(shape,dtype))->tuple[Tensor(shape,'uint8'),Tensor(scales_shape,'float32')]:
        out=output(shape,'uint8');scales=output(scales_shape,'float32')
        with T.Kernel(T.ceildiv(rows*(k//128),groups),threads=groups*32) as block:
            tx=T.get_thread_binding();lane=tx%32;group=block*groups+tx//32
            words=buffer_view(out,shape=(rows*k//4,),dtype='uint32')
            metadata=buffer_view(scales,shape=(rows*(k//128),),dtype='float32')
            if group<rows*(k//128):
                values=T.alloc_local((4,),'float32')
                maximum=T.alloc_var('float32');maximum=0
                for j in T.unroll(4):
                    row=group//(k//128);tile=group%(k//128)
                    if top:
                        values[j]=T.cast(x[row//top,row%top,tile*128+lane*4+j],'float32')
                    else:
                        values[j]=T.cast(x[row,tile*128+lane*4+j],'float32')
                    maximum=T.max(maximum,T.abs(values[j]))
                for shift in T.unroll(5):
                    maximum=T.max(maximum,T.shfl_down(maximum,16>>shift))
                maximum=T.shfl_sync(maximum,0)
                scale=T.max(maximum,T.float32(1e-12))/T.float32(448)
                bits=T.call_extern('uint32','tensor_encode_e4m3x4',
                    values[0]/scale,values[1]/scale,values[2]/scale,values[3]/scale)
                words[group*32+lane]=bits
                if lane==0:metadata[group]=scale
        return out,scales
    return compute
