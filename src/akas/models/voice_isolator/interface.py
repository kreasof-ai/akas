"""Open the compiled Voice Isolator for offline audio enhancement."""

from contextlib import contextmanager
import json
from pathlib import Path

from akas.models.voice_isolator.model import VoiceIsolator
from akas.tensor.device import vulkan_device


@contextmanager
def service(checkpoint, bundle, *, device=0, compact_scratch=True):
    """Expose isolation with the weight precision selected at compilation."""
    filename, _ = VoiceIsolator.forward_index
    index = json.loads((Path(bundle) / filename).read_text(encoding="utf-8"))
    loaded = VoiceIsolator.checkpoint_type(
        checkpoint,
        weight_dtype=index["weight_dtype"],
        fp32_projections=index.get("fp32_projections", ()),
    )
    with (
        vulkan_device(device) as adapter,
        VoiceIsolator(
            loaded, bundle, adapter, compact_scratch=compact_scratch
        ) as model,
    ):
        yield model
