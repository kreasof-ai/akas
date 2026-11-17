"""Serve complete PCM utterances while keeping device work on one owner thread."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SpeechService:
    synthesize: Callable
    metadata: dict
    languages: tuple[str, ...]


def create_app(factory):
    from aiohttp import web
    import numpy as np

    service = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="akas-speech")

    async def lifecycle(app):
        nonlocal service
        loop = asyncio.get_running_loop()
        context = None
        try:

            def open_service():
                scope = factory()
                return scope, scope.__enter__()

            context, service = await loop.run_in_executor(executor, open_service)
            yield
        finally:
            if context is not None:
                await loop.run_in_executor(executor, context.__exit__, None, None, None)
            executor.shutdown()

    async def health(request):
        return web.json_response({"status": "ok"})

    async def props(request):
        return web.json_response(service.metadata)

    async def tts(request):
        try:
            payload = await request.json()
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("input"), str)
                or not payload["input"].strip()
                or payload.get("response_format") != "pcm16"
                or payload.get("stream", False)
            ):
                raise ValueError("requires text and non-streaming PCM16")
            language = payload.get("lang", "en")
            if language not in service.languages:
                raise ValueError("unsupported speech language")
            options = {
                name: payload[key]
                for key, name in (
                    ("voice", "voice"),
                    ("seed", "seed"),
                    ("temperature", "temperature"),
                    ("n_predict", "limit"),
                )
                if key in payload
            }

            def synthesize():
                return service.synthesize(payload["input"], lang=language, **options)

            speech = await asyncio.get_running_loop().run_in_executor(
                executor, synthesize
            )
        except (ValueError, TypeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        pcm = np.rint(speech.samples * 32767).astype("<i2").tobytes()
        return web.Response(
            body=pcm,
            headers={
                "Content-Type": f"audio/pcm;rate={speech.sample_rate};encoding=signed-int;bits=16",
                "X-AKAS-Audio-Frames": str(speech.frames),
                "X-AKAS-Terminated": str(speech.terminated).lower(),
            },
        )

    app = web.Application(client_max_size=1024**2)
    app.cleanup_ctx.append(lifecycle)
    app.add_routes(
        [web.get("/health", health), web.get("/props", props), web.post("/tts", tts)]
    )
    return app
