"""Expose the compiled Qwen3 ASR model as a transcription service."""

from contextlib import contextmanager
from pathlib import Path

from akas.models.qwen3_asr.model import Qwen3ASR


@contextmanager
def service(checkpoint, projector, bundle, *, device=0):
    """Open transcription resources on the calling thread."""
    from akas.benchmarking.provenance import digest
    from akas.serving.speech import TranscriptionService
    from akas.tensor.device import vulkan_device

    loaded = Qwen3ASR.checkpoint_type(checkpoint, projector)
    with vulkan_device(device) as adapter, Qwen3ASR(loaded, bundle, adapter) as model:
        yield TranscriptionService(
            model.transcribe,
            dict(
                device=dict(adapter.info, name=adapter.info["adapter"]["device"]),
                program_sha256=digest(Path(bundle) / model.forward_index[0]),
                **model.runtime_metadata(),
            ),
        )
