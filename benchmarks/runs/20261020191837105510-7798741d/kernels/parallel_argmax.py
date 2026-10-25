"""Exact comparison-only vocabulary reduction, preserving first-index ties."""
from akas.kernels.kernel import Tensor, kernel, output


def partial(spec, schedule):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce
    p = {**spec.parameters, **schedule}
    rows, columns, block = p['r'], p['c'], p['block']
    parts = (columns + block - 1) // block
    maximum = tree_reduce(128, 'max')

    @kernel
    def compute(x: Tensor((rows * columns,), 'float32')) -> tuple[Tensor((rows, parts), 'float32'), Tensor((rows, parts), 'int32')]:
        values = output((rows, parts), 'float32')
        indices = output((rows, parts), 'int32')
        with T.Kernel(parts, rows, threads=128) as (part, row):
            tx = T.get_thread_binding()
            best = T.alloc_var('float32')
            index = T.alloc_var('int32')
            best = -T.infinity('float32')
            index = columns
            scratch = T.alloc_shared((128,), 'float32')
            winners = T.alloc_shared((128,), 'int32')
            for tile in T.serial(T.ceildiv(block, 128)):
                col = part * block + tile * 128 + tx
                if col < columns and tile * 128 + tx < block:
                    if x[row * columns + col] > best:
                        best = x[row * columns + col]
                        index = col
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            peak = T.alloc_var('float32')
            peak = scratch[0]
            winners[tx] = T.if_then_else(best == peak, -index, -columns)
            T.sync_threads()
            maximum(winners, tx)
            if tx == 0:
                values[row, part] = peak
                indices[row, part] = -winners[0]
        return values, indices

    return compute


def merge(spec, schedule):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce
    p = {**spec.parameters, **schedule}
    rows, columns, block = p['r'], p['c'], p['block']
    parts = (columns + block - 1) // block
    maximum = tree_reduce(128, 'max')

    @kernel
    def compute(values: Tensor((rows, parts), 'float32'), indices: Tensor((rows, parts), 'int32'), enabled: Tensor((rows,), 'int32')) -> Tensor((rows,), 'int32'):
        out = output((rows,), 'int32')
        with T.Kernel(rows, threads=128) as row:
            tx = T.get_thread_binding()
            best = T.alloc_var('float32')
            index = T.alloc_var('int32')
            best = -T.infinity('float32')
            index = columns
            scratch = T.alloc_shared((128,), 'float32')
            winners = T.alloc_shared((128,), 'int32')
            for tile in T.serial(T.ceildiv(parts, 128)):
                part = tile * 128 + tx
                if part < parts:
                    value = values[row, part]
                    candidate = indices[row, part]
                    if value > best or (value == best and candidate < index):
                        best = value
                        index = candidate
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            peak = T.alloc_var('float32')
            peak = scratch[0]
            winners[tx] = T.if_then_else(best == peak, -index, -columns)
            T.sync_threads()
            maximum(winners, tx)
            if tx == 0 and enabled[row] != 0:
                out[row] = -winners[0]
        return out

    return compute
