"""HTTP streaming over an Engine; factories own model and device setup."""

import asyncio
from dataclasses import dataclass, field
import json
from threading import Event, Thread
from typing import Callable

from akas.engine import Engine


@dataclass
class Service:
    """Yield from a factory context manager on the engine's owner thread.

    encode returns plain token IDs without adding special tokens. Factories
    supply truthful model metadata and close resources when the context exits.
    """

    engine: Engine
    encode: Callable[[str], list[int]] | None = None
    metadata: dict = field(default_factory=dict)


class _Worker:
    def __init__(self, factory):
        self.factory = factory
        self.service = None
        self.failure = None
        self.requests = {}
        self.stopped = Event()

    async def start(self, app):
        loop = asyncio.get_running_loop()
        ready = loop.create_future()

        def publish(events):
            def deliver():
                for request, event in events:
                    if not request.cancelled.is_set():
                        request.channel.put_nowait(event)

            loop.call_soon_threadsafe(deliver)

        def initialized():
            if not ready.done():
                ready.set_result(None)

        def failed(error):
            self.failure = str(error)
            if not ready.done():
                ready.set_exception(error)
            for request in self.requests.values():
                request.channel.put_nowait(dict(error=str(error)))

        def run():
            try:
                with self.factory() as service:
                    self.service = service
                    engine = service.engine
                    engine.scheduler.publish = publish
                    try:
                        if self.stopped.is_set():
                            return
                        loop.call_soon_threadsafe(initialized)
                        engine.run()
                    finally:
                        engine.close()
            except BaseException as error:
                loop.call_soon_threadsafe(failed, error)

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()
        try:
            await ready
        except BaseException:
            await self.close(app)
            raise

    async def close(self, app):
        self.stopped.set()
        for request in self.requests.values():
            request.cancelled.set()
            request.channel.put_nowait(dict(error="server stopped"))
        if self.service is not None:
            self.service.engine.scheduler.stopped.set()
        await asyncio.to_thread(self.thread.join)


def create_app(factory):
    """Serve any factory yielding Service, using the shared Engine scheduler.

    Model resources are created, run and closed on one worker thread. HTTP
    handlers only enqueue requests and consume events on the asyncio loop.
    Generation is greedy and length bounded; callers explicitly ignore EOS.
    """
    from aiohttp import web

    worker = _Worker(factory)
    app = web.Application(client_max_size=8 * 1024**2)
    app.on_startup.append(worker.start)
    app.on_shutdown.append(worker.close)

    def available():
        if worker.failure or worker.stopped.is_set():
            raise web.HTTPServiceUnavailable(text=worker.failure or "server stopped")
        return worker.service

    async def payload(request):
        try:
            data = await request.json()
        except (ValueError, UnicodeError):
            raise web.HTTPBadRequest(text="requires a JSON object")
        if not isinstance(data, dict):
            raise web.HTTPBadRequest(text="requires a JSON object")
        return data

    async def health(request):
        failed = bool(worker.failure or worker.stopped.is_set())
        return web.json_response(
            dict(status="failed" if failed else "ok"), status=503 if failed else 200
        )

    async def info(request):
        service = available()
        engine = service.engine
        model = engine.model
        return web.json_response(
            {
                **service.metadata,
                "engine": "tensor",
                "capacity": model.capacity,
                "context": model.context,
                "vocab": model.config.vocab,
                "speculative": hasattr(model, "decode_many"),
                "stats": engine.scheduler.stats,
                "speculation_stats": getattr(model, "stats", None),
            }
        )

    async def tokenize(request):
        service = available()
        if service.encode is None:
            raise web.HTTPNotImplemented(text="this service accepts token IDs only")
        data = await payload(request)
        if not isinstance(data.get("prompt"), str):
            raise web.HTTPBadRequest(text="prompt must be a string")
        return web.json_response(dict(tokens=service.encode(data["prompt"])))

    async def generate(request):
        service = available()
        data = await payload(request)
        sampling = data.get("sampling_params", {})
        if (
            not isinstance(sampling, dict)
            or sampling.get("temperature", 0) != 0
            or sampling.get("ignore_eos") is not True
        ):
            raise web.HTTPBadRequest(
                text="requires greedy generation with ignore_eos=true"
            )
        if not isinstance(data.get("input_ids"), list):
            raise web.HTTPBadRequest(text="input_ids must be a list of token IDs")
        try:
            item = service.engine.submit(
                data.get("input_ids"), max_tokens=sampling.get("max_new_tokens", 256)
            )
        except (ValueError, TypeError) as error:
            raise web.HTTPBadRequest(text=str(error))
        except RuntimeError as error:
            raise web.HTTPServiceUnavailable(text=str(error))
        item.channel = asyncio.Queue()
        worker.requests[id(item)] = item
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        try:
            await response.prepare(request)
            while True:
                event = await item.channel.get()
                await response.write(
                    (
                        "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
                    ).encode()
                )
                if event.get("error") or event.get("meta_info", {}).get(
                    "finish_reason"
                ):
                    break
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        finally:
            item.cancelled.set()
            worker.requests.pop(id(item), None)

    app.router.add_get("/health", health)
    app.router.add_get("/server_info", info)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/generate", generate)
    return app
