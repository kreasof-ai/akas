"""Shared concurrent token-arrival protocol for engine and baseline streams."""

def phase_metrics(batch, prefix, output, first, decode, *, input_tokens=None):
    """Burst rates from request timings, including scheduling; not kernel timings."""
    import math

    if (
        type(batch) is not int or batch < 1 or type(output) is not int or output < 2
        or len(first) != batch or len(decode) != batch
        or any(not math.isfinite(v) or v <= 0 for v in (*first, *decode))
    ):
        raise ValueError("runtime did not expose valid phase timings")
    return dict(
        phase_protocol="request-phase-v1",
        prefill_tokens_per_second=(batch * prefix if input_tokens is None else input_tokens) / max(first),
        decode_tokens_per_second=batch * (output - 1) / max(decode),
        ttft_seconds=max(first),
        tpot_seconds=max(decode) / (output - 1),
    )


async def stream_burst(prompts, generate, output_tokens):
    """Measure concurrent cumulative-token streams from one burst arrival.

    Adapters yield (cumulative token IDs, metadata). Observed arrival times
    include queueing and delivery. Coalesced tokens share an arrival timestamp;
    the retained counts expose coalescing instead of inventing token timings.
    """
    import asyncio
    import time

    if not prompts or any(not prompt for prompt in prompts) or type(output_tokens) is not int or output_tokens < 2:
        raise ValueError("streamed phase measurements need nonempty prompts and at least two outputs")
    started = time.perf_counter()

    async def request(index, prompt):
        tokens, arrivals, chunks, metadata = [], [], [], {}
        async for current, metadata in generate(index, prompt):
            now = time.perf_counter() - started
            current = list(current)
            if current[:len(tokens)] != tokens or len(current) > output_tokens:
                raise ValueError("stream changed committed tokens or exceeded output limit")
            added = len(current) - len(tokens)
            if added:
                arrivals.extend([now] * added)
                chunks.append(dict(seconds=now, tokens=added))
            tokens = current
        if len(tokens) != output_tokens:
            raise ValueError("stream returned an incomplete output")
        if metadata.get("cached_tokens", 0) != 0:
            raise ValueError("prefix reuse violates the empty-cache workload")
        return tokens, dict(arrivals_seconds=arrivals, chunks=chunks, metadata=metadata)

    tasks = [asyncio.create_task(request(i, prompt)) for i, prompt in enumerate(prompts)]
    try:
        rows = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    timings = [details for _, details in rows]
    phases = phase_metrics(len(prompts), len(prompts[0]), output_tokens,
                          [r["arrivals_seconds"][0] for r in timings],
                          [r["arrivals_seconds"][-1] - r["arrivals_seconds"][0] for r in timings],
                          input_tokens=sum(map(len, prompts)))
    phases["phase_protocol"] = "client-stream-burst-v1"
    phases["request_streams"] = timings
    return [tokens for tokens, _ in rows], phases


async def engine_burst(engine, prompts, output_tokens):
    """Borrow an idle Engine and measure its real scheduler token deliveries."""
    import asyncio

    engine._check_thread()
    scheduler = engine.scheduler
    if (engine.closed or scheduler.pending or scheduler.prefilling or scheduler.decoding
            or not scheduler.incoming.empty()):
        raise ValueError("streamed benchmark requires an idle, open engine")
    queues, requests = {}, []
    previous = scheduler.publish

    def publish(events):
        for request, row in events:
            queues[id(request)].put_nowait(row)

    scheduler.publish = publish

    async def generate(index, prompt):
        request = engine.submit(prompt, max_tokens=output_tokens)
        requests.append(request)
        queue = queues[id(request)] = asyncio.Queue()
        try:
            while True:
                row = await queue.get()
                if isinstance(row, Exception):
                    raise row
                yield row["output_ids"], row["meta_info"]
                if row["meta_info"]["finish_reason"] is not None:
                    return
        finally:
            if request.slot is not None or not request.output:
                engine.cancel(request)

    async def drive():
        try:
            while True:
                # Allow all arrivals and token consumers to run between turns.
                await asyncio.sleep(0)
                engine.step()
        except Exception as error:
            for queue in queues.values():
                queue.put_nowait(error)

    driver = asyncio.create_task(drive())
    try:
        return await stream_burst(prompts, generate, output_tokens)
    finally:
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        for request in requests:
            if request.slot is not None or len(request.output) < request.limit:
                engine.cancel(request)
        # Flush cancellation ownership before restoring the caller's publisher.
        try:
            engine.step()
        finally:
            scheduler.publish = previous
