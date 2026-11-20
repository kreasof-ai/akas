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


@dataclass(frozen=True)
class TranscriptionService:
    transcribe: Callable
    metadata: dict


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

    async def transcribe(request):
        from base64 import b64decode

        if not isinstance(service, TranscriptionService):
            raise web.HTTPNotImplemented(text="transcription requires an ASR service")
        try:
            payload = await request.clone(client_max_size=16 * 1024**2).json()
            if (
                payload.get("stream", False)
                or payload.get("temperature", 0) != 0
                or len(payload["messages"]) != 1
            ):
                raise ValueError("requires a single non-streaming greedy audio request")
            message = payload["messages"][0]
            parts = message["content"]
            if (
                message["role"] != "user"
                or len(parts) != 1
                or parts[0]["type"] != "input_audio"
            ):
                raise ValueError("requires one user input_audio part")
            audio = parts[0]["input_audio"]
            if audio["format"] != "wav":
                raise ValueError("requires WAV audio")
            wav = b64decode(audio["data"], validate=True)
            language = payload["chat_template_kwargs"]["language"]
            limit = payload.get("max_tokens", 128)

            def recognize():
                return service.transcribe(wav, language=language, limit=limit)

            result = await asyncio.get_running_loop().run_in_executor(
                executor, recognize
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop" if result.terminated else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": len(result.token_ids),
                    "total_tokens": result.prompt_tokens + len(result.token_ids),
                },
                "token_ids": result.token_ids,
            }
        )

    async def tts(request):
        if isinstance(service, TranscriptionService):
            raise web.HTTPNotImplemented(text="synthesis requires a TTS service")
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

    async def speech_stream(request):
        if not hasattr(service, "engine"):
            raise web.HTTPNotImplemented(
                text="streaming speech requires a streaming service"
            )
        try:
            payload = await request.json()
        except (ValueError, UnicodeError) as error:
            raise web.HTTPBadRequest(text="requires a JSON object") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="requires a JSON object")
        text = payload.get("input") or payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise web.HTTPBadRequest(text="input must be a non-empty string")
        fmt = payload.get("response_format", payload.get("format", "pcm"))
        if fmt != "pcm":
            raise web.HTTPBadRequest(
                text="streaming /v1/audio/speech requires pcm format"
            )
        decoder = payload.get("decoder") or "keep2_plain"
        if decoder != "keep2_plain":
            raise web.HTTPBadRequest(text=f"unsupported decoder: {decoder}")
        stream_decode = payload.get("stream_decode") or "prefix"
        if stream_decode not in ("prefix", "whole"):
            raise web.HTTPBadRequest(text=f"unsupported stream_decode: {stream_decode}")
        seed = payload.get("seed")
        scope = payload.get("stream_codebooks_scope", "first")
        if scope not in ("first", "all"):
            raise web.HTTPBadRequest(text="invalid stream_codebooks_scope")

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def on_chunk(data):
            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", data))

        def on_done():
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        def on_error(msg):
            loop.call_soon_threadsafe(queue.put_nowait, ("error", msg))

        try:
            job = service.engine.submit(
                text,
                seed=1000 if seed is None else seed,
                temperature=payload.get("temperature", 0.75),
                max_tokens=payload.get("max_tokens", 384),
                stream_decode=stream_decode,
                stream_codebooks=payload.get("stream_codebooks", 1),
                first_only=scope == "first",
                on_chunk=on_chunk,
                on_done=on_done,
                on_error=on_error,
            )
        except (ValueError, TypeError, OverflowError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        response = None
        sample_rate = int(service.metadata.get("sample_rate_hz", 44100))
        headers = {
            "Content-Type": "audio/pcm",
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Format": "float32",
            "X-Seed": str(job.seed),
            "X-Decoder": decoder,
        }
        try:
            while True:
                kind, item = await queue.get()
                if kind == "error":
                    if response is None:
                        raise web.HTTPBadRequest(text=str(item))
                    break
                if kind == "chunk" and item:
                    if response is None:
                        response = web.StreamResponse(status=200, headers=headers)
                        await response.prepare(request)
                    await response.write(item)
                elif kind == "done":
                    if response is None:
                        response = web.StreamResponse(status=200, headers=headers)
                        await response.prepare(request)
                    await response.write_eof()
                    return response
        except asyncio.CancelledError:
            job.cancelled.set()
            raise
        finally:
            job.cancelled.set()
        return response

    app = web.Application(client_max_size=1024**2)
    app.cleanup_ctx.append(lifecycle)
    app.add_routes(
        [
            web.get("/health", health),
            web.get("/props", props),
            web.post("/tts", tts),
            web.post("/v1/audio/speech", speech_stream),
            web.post("/v1/chat/completions", transcribe),
        ]
    )
    return app
