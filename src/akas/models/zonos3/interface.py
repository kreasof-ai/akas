"""Expose compiled ZONOS3-Mini for complete and streaming speech synthesis."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING

from akas.benchmarking.provenance import digest
from akas.models.zonos3.config import LANGUAGES
from akas.models.zonos3.model import Zonos3MiniSynthesis
from akas.tensor.device import vulkan_device

if TYPE_CHECKING:
    from akas.models.zonos3.streaming import Zonos3StreamingEngine


@dataclass(frozen=True)
class StreamingSpeechService:
    engine: "Zonos3StreamingEngine"
    metadata: dict


@contextmanager
def service(checkpoint, projector, bundle, *, device=0):
    """Open a compiled model on the speech server's device-owner thread."""
    from akas.serving.speech import SpeechService

    loaded = Zonos3MiniSynthesis.checkpoint_type(checkpoint, projector)
    with (
        vulkan_device(device) as adapter,
        Zonos3MiniSynthesis(loaded, bundle, adapter) as model,
    ):
        name = adapter.info["adapter"]["device"]
        yield SpeechService(
            model.synthesize,
            dict(
                device=dict(adapter.info, name=name),
                program_sha256=digest(Path(bundle) / model.forward_index[0]),
                sample_rate_hz=loaded.config.sample_rate,
                default_voices=loaded.default_voices,
                fallback_voice=loaded.projector.metadata.get(
                    "clip.gen.audio.default_voice"
                ),
            ),
            languages=LANGUAGES,
        )


@contextmanager
def streaming_service(
    checkpoint_dir, codec_path, bundle_dir, *, device=0, max_requests=256
):
    """Open the CUDA streaming runtime and run its request scheduler."""
    import tensor

    from akas.models.zonos3.streaming import (
        STREAM_INDEX,
        Zonos3StreamingEngine,
        Zonos3StreamingRuntime,
    )
    from akas.models.zonos3.weights import Zonos3MiniCheckpoint

    loaded = Zonos3MiniCheckpoint(checkpoint_dir, codec_path)
    with (
        tensor.Device(device) as adapter,
        Zonos3StreamingRuntime(
            loaded, bundle_dir, adapter, max_requests=max_requests
        ) as runtime,
    ):
        engine = Zonos3StreamingEngine(runtime)
        worker = Thread(
            target=engine.run_forever, daemon=True, name="akas-zonos3-stream"
        )
        worker.start()
        try:
            yield StreamingSpeechService(
                engine=engine,
                metadata={
                    "engine": "tensor",
                    "provider": adapter.info["provider"],
                    "device": dict(adapter.info),
                    "program_sha256": digest(Path(bundle_dir) / STREAM_INDEX[0]),
                    "sample_rate_hz": loaded.config.sample_rate,
                    "max_requests": runtime.max_requests,
                    "pool_capacity": runtime.pool_capacity,
                    "context": runtime.context,
                    "decoder": "keep2_plain",
                },
            )
        finally:
            engine.stop()
            worker.join()
