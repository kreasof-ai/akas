"""Token arrival measurements preserve concurrency, coalescing and failures."""

import asyncio
import pytest

from tools.baselines.common import stream_burst


def test_streamed_burst_retains_each_request_and_delivery_chunks():
    async def generate(index, prompt):
        await asyncio.sleep(.001 * (index + 1))
        yield [prompt[0]], {"cached_tokens": 0}
        await asyncio.sleep(.001)
        yield [prompt[0], 7, 8], {"cached_tokens": 0}

    output, details = asyncio.run(stream_burst([[1, 2], [3, 4]], generate, 3))
    assert output == [[1, 7, 8], [3, 7, 8]]
    assert details["phase_protocol"] == "client-stream-burst-v1"
    times = [r["arrivals_seconds"] for r in details["request_streams"]]
    assert all(row[0] < row[1] == row[2] for row in times)
    assert details["ttft_seconds"] == max(row[0] for row in times)
    assert details["prefill_tokens_per_second"] == 4 / details["ttft_seconds"]
    assert details["decode_tokens_per_second"] == 4 / max(row[-1] - row[0] for row in times)


@pytest.mark.parametrize("problem", ["rewrite", "cache", "short"])
def test_failed_stream_cancels_other_requests(problem):
    closed = []

    async def generate(index, prompt):
        try:
            yield [1], {}
            if index == 0:
                await asyncio.sleep(.001)
                if problem == "short":
                    return
                yield [9, 2] if problem == "rewrite" else [1, 2], {"cached_tokens": int(problem == "cache")}
            else:
                await asyncio.sleep(10)
        finally:
            closed.append(index)

    with pytest.raises(ValueError):
        asyncio.run(stream_burst([[1], [2]], generate, 2))
    assert sorted(closed) == [0, 1]


def test_engine_streams_use_the_same_protocol_and_release_slots():
    from akas import Engine, EngineConfig
    from akas.benchmarking.streaming import engine_burst
    from test_engine import Adapter

    model = Adapter(2)
    with Engine(model, config=EngineConfig(prefill_chunk=2, prefill_token_budget=3)) as engine:
        previous = engine.scheduler.publish
        output, details = asyncio.run(engine_burst(engine, [[1, 2, 3], [8]], 5))
        assert output == [[6, 12, 24, 48, 96], [8, 16, 32, 64, 28]]
        assert details["phase_protocol"] == "client-stream-burst-v1"
        assert details["prefill_tokens_per_second"] == 4 / details["ttft_seconds"]
        assert engine.scheduler.publish is previous
        assert not model.state and not engine.scheduler.decoding
        assert asyncio.run(engine_burst(engine, [[2]], 2))[0] == [[2, 4]]


def test_engine_failure_unblocks_streams_and_releases_all_requests():
    from akas import Engine
    from akas.benchmarking.streaming import engine_burst
    from test_engine import Adapter

    model = Adapter(2)
    def fail(*args):
        raise RuntimeError("device forward failed")
    model.decode = fail
    with Engine(model) as engine:
        previous = engine.scheduler.publish
        async def measure():
            return await asyncio.wait_for(engine_burst(engine, [[1], [2]], 5), timeout=2)
        with pytest.raises(RuntimeError, match="device forward failed"):
            asyncio.run(measure())
        assert not model.state
        assert engine.scheduler.publish is previous
