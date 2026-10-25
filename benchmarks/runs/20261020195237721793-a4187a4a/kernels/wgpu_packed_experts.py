"""Research expert GEMV consuming a quantization-owned packed lane product."""
from akas.kernels.kernel import Tensor, kernel, output


def factory(spec, schedule):
    import tilelang.language as T
    from wgpu_q4_dot import bind

    p = {**spec.parameters, **schedule}
    depth, columns, experts, top = (p[n] for n in ('k', 'o', 'experts', 'top'))
    width, unroll = p.get('columns_per_warp', 1), p.get('unroll', 1)
    if p['r'] != 1 or p.get('input_dtype') != 'float32' or p.get('round_inputs', False):
        raise ValueError('Packed F32 expert decode needs one F32 row')
    codec = bind(p['weight_parts'], depth)
    storage, dot = codec.storage.values, codec.require('dot')
    routed = p.get('routed_input', False)
    shape = (1, top, depth) if routed else (1, depth)

    @kernel
    def compute(x: Tensor(shape, 'float32'), w: Tensor(storage.shape, storage.dtype),
                ids: Tensor((1, top), 'int32'), counts: Tensor((experts,), 'int32')) -> Tensor((1, top, columns), 'float32'):
        out = output((1, top, columns), 'float32')
        flat = T.meta_var(T.view(x, (top * depth if routed else depth,), dtype='float32'))
        with T.Kernel(T.ceildiv(columns, 4 * width), top, threads=128) as (bx, rank):
            tx = T.get_thread_binding()
            lane, warp = tx % 32, tx // 32
            expert = ids[0, rank]
            scratch = T.alloc_shared((width, 128), 'float32')
            accum = T.alloc_local((width,), 'float32')
            for col in T.unroll(width):
                accum[col] = 0
            for tile in T.serial(T.ceildiv(depth // 256, unroll)):
                for step in T.unroll(unroll):
                    for col in T.unroll(width):
                        column = (bx * 4 + warp) * width + col
                        if column < columns and tile * unroll + step < depth // 256:
                            accum[col] += dot((w,), flat, rank * depth if routed else 0,
                                              tile * unroll + step, lane, expert * columns + column)
            for col in T.unroll(width):
                if T.call_extern('uint32', 'tensor_subgroup_size') >= 32:
                    for step in T.unroll(5):
                        accum[col] += T.call_extern('float32', 'subgroupShuffleXor', accum[col], T.uint32(16 >> step))
                else:
                    scratch[col, tx] = accum[col]
                    T.sync_threads()
                    for step in T.unroll(5):
                        if lane < 16 >> step:
                            scratch[col, tx] += scratch[col, tx + (16 >> step)]
                        T.sync_threads()
                    accum[col] = scratch[col, tx]
                column = (bx * 4 + warp) * width + col
                if lane == 0 and column < columns:
                    out[0, rank, column] = T.if_then_else(counts[expert] > 0, accum[col], 0)
        return out

    return compute
