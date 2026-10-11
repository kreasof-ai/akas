"""Exercise shared HTTP transport against independent and native-batch adapters."""

import asyncio
from contextlib import contextmanager
import json
from threading import get_ident
import time
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
import pytest

from akas import Engine, EngineConfig
from akas.serving import Service
from akas.serving import create_app


class Model:
    capacity = 2
    context = 64

    def __init__(self, vocab, *, failure=False, delay=0):
        self.config = SimpleNamespace(vocab=vocab)
        self.thread = get_ident()
        self.owned = set()
        self.released = []
        self.closed = False
        self.failure = failure
        self.delay = delay

    def check(self):
        assert self.thread == get_ident()

    def admit(self):
        self.check()
        slot = next(i for i in range(self.capacity) if i not in self.owned)
        self.owned.add(slot)
        return slot

    def release(self, slot):
        self.check()
        self.owned.remove(slot)
        self.released.append(slot)

    def decode(self, slots, tokens):
        self.check()
        time.sleep(self.delay)
        if self.failure:
            raise RuntimeError("injected inference failure")
        return [(token + 1) % self.config.vocab for token in tokens]

    def close(self):
        self.check()
        assert not self.owned
        self.closed = True


class Independent(Model):
    def prefill_batch(self, requests, sequences, chunk):
        return self.decode([r.slot for r in requests], [s[-1] for s in sequences])


class Batched(Model):
    def batch(self, count, chunk):
        return SimpleNamespace(
            run=lambda slots, sequences: self.decode(slots, [s[-1] for s in sequences])
        )


def factory(kind, created, *, encoder=True, failure=False, delay=0):
    @contextmanager
    def load():
        model = kind(19 if kind is Independent else 73, failure=failure, delay=delay)
        created.append(model)
        with Engine(model, config=EngineConfig(prefill_chunk=2)) as engine:
            yield Service(
                engine,
                (lambda text: [len(text)]) if encoder else None,
                {"model": kind.__name__},
            )

    return load


def body(prompt, count=5):
    return dict(
        input_ids=prompt,
        sampling_params=dict(max_new_tokens=count, temperature=0, ignore_eos=True),
    )


async def events(response):
    assert response.status == 200
    assert response.headers["Content-Type"] == "text/event-stream"
    text = await asyncio.wait_for(response.text(), 5)
    assert text.endswith("data: [DONE]\n\n")
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("kind", [Independent, Batched])
def test_shared_http_streams_metadata_tokenization_and_owner_thread_cleanup(kind):
    created = []

    async def check():
        async with TestClient(TestServer(create_app(factory(kind, created)))) as client:
            assert (await client.get("/health")).status == 200
            info = await (await client.get("/server_info")).json()
            assert info["model"] == kind.__name__
            assert info["vocab"] == created[0].config.vocab
            assert info["capacity"] == 2 and info["context"] == 64
            assert (
                await (await client.post("/tokenize", json={"prompt": "abc"})).json()
            ) == {"tokens": [3]}
            responses = await asyncio.gather(
                client.post("/generate", json=body([1, 2, 3])),
                client.post("/generate", json=body([7], 2)),
            )
            first, second = await asyncio.gather(*(events(r) for r in responses))
            assert first[-1]["output_ids"] == [4, 5, 6, 7, 8]
            assert second[-1]["output_ids"] == [8, 9]
            assert first[-1]["meta_info"]["prompt_tokens"] == 3
            assert first[-1]["meta_info"]["completion_tokens"] == 5
            assert first[-1]["meta_info"]["finish_reason"]["type"] == "length"
            assert first[0]["output_ids"] == [4]
            assert not created[0].owned
        assert created[0].closed and created[0].thread != get_ident()

    asyncio.run(check())


@pytest.mark.parametrize("kind", [Independent, Batched])
def test_http_validates_actual_model_limits_and_malformed_payloads(kind):
    created = []

    async def check():
        async with TestClient(
            TestServer(create_app(factory(kind, created, encoder=False)))
        ) as client:
            vocab = created[0].config.vocab
            invalid = [
                body([vocab]),
                body([-1]),
                body([True]),
                body([]),
                body([1], 0),
                body([1], True),
                body([1] * 64),
                body({"1": 2}),
                body("12"),
                {"input_ids": [1], "sampling_params": []},
                [],
                {"input_ids": [1], "sampling_params": {"ignore_eos": False}},
                {
                    "input_ids": [1],
                    "sampling_params": {"ignore_eos": True, "temperature": 1},
                },
            ]
            for data in invalid:
                assert (await client.post("/generate", json=data)).status == 400, data
            assert (await client.post("/generate", data="{")).status == 400
            assert (
                await client.post("/tokenize", json={"prompt": "abc"})
            ).status == 501
            good = await events(
                await client.post("/generate", json=body([vocab - 1], 1))
            )
            assert good[-1]["output_ids"] == [0]
        assert created[0].closed

    asyncio.run(check())


def test_worker_failure_reaches_waiting_stream_and_health():
    created = []

    async def check():
        async with TestClient(
            TestServer(create_app(factory(Independent, created, failure=True)))
        ) as client:
            response = await client.post("/generate", json=body([1]))
            result = await events(response)
            assert result[-1]["error"] == "injected inference failure"
            assert (await client.get("/health")).status == 503
            assert (await client.post("/generate", json=body([1]))).status == 503
        assert created[0].closed

    asyncio.run(check())


def test_disconnected_stream_releases_request_without_stopping_engine():
    created = []

    async def check():
        server = TestServer(
            create_app(factory(Independent, created, delay=0.01)),
            handler_cancellation=True,
        )
        async with TestClient(server) as client:
            response = await client.post("/generate", json=body([1], 50))
            assert (await response.content.readline()).startswith(b"data: ")
            response.close()
            for _ in range(100):
                if created[0].released:
                    break
                await asyncio.sleep(0.01)
            assert not created[0].owned and created[0].released
            assert (await client.get("/health")).status == 200
            assert (await events(await client.post("/generate", json=body([2], 1))))[
                -1
            ]["output_ids"] == [3]
        assert created[0].closed

    asyncio.run(check())


def test_startup_failure_closes_factory_resources():
    closed = []

    @contextmanager
    def broken():
        try:
            raise ValueError("bad checkpoint")
            yield
        finally:
            closed.append(True)

    async def check():
        client = TestClient(TestServer(create_app(broken)))
        try:
            with pytest.raises(ValueError, match="bad checkpoint"):
                await client.start_server()
        finally:
            await client.close()
        assert closed

    asyncio.run(check())


def test_speculative_adapter_exposes_target_limits_for_engine_submission(monkeypatch):
    import akas.execution.bindings.bf16.speculative as mtp

    closed = []
    target = SimpleNamespace(
        capacity=2,
        context=64,
        config=SimpleNamespace(vocab=73),
        close=lambda: closed.append("target"),
    )
    monkeypatch.setattr(
        mtp,
        "DenseMTP",
        lambda model: SimpleNamespace(close=lambda: closed.append("draft")),
    )
    with Engine(mtp.SpeculativeEngine(target)) as engine:
        assert engine.submit([72], max_tokens=1).prompt == [72]
        with pytest.raises(ValueError, match="token IDs"):
            engine.submit([73], max_tokens=1)
    assert closed == ["draft", "target"]
