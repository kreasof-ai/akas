"""Shared TileLang computations for moe."""


def quantized_routes(spec, schedule):
    """Deterministic compact token routes, shared independently of weight format."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, experts, top = (p["r"], p["experts"], p["top"])

    @kernel
    def compute(
        ids: Tensor((rows, top), "int32"), active: Tensor((rows,), "int32")
    ) -> tuple[Tensor((experts,), "int32"), Tensor((experts, rows), "int32")]:
        counts = output((experts,), "int32")
        routes = output((experts, rows), "int32")
        with T.Kernel(experts, threads=32) as expert:
            for row in T.Parallel(rows):
                routes[expert, row] = -1
            T.sync_threads()
            for lane in T.Parallel(1):
                count = T.alloc_var("int32")
                count = 0
                for row in T.serial(rows):
                    if active[row] != 0:
                        for rank in T.serial(top):
                            if ids[row, rank] == expert:
                                routes[expert, count] = row * top + rank
                                count += 1
                counts[expert] = count
        return (counts, routes)

    return compute
