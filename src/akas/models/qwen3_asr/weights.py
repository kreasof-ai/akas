"""Read and bind the pinned Qwen3 ASR GGUF pair."""

from collections import Counter
from contextlib import ExitStack
from math import prod

import numpy as np

from akas.checkpoints.gguf import GGUF
from akas.models.qwen3_asr.config import Qwen3ASRConfig

TOKEN_EMBEDDING = "token_embd.weight"
CONV_PROJECTION = "a.conv_out.weight"
HALF_SUFFIX = ".f16"
TAP_SUFFIX = ".taps"
ALIGNED_SUFFIX = ".q8_words"


class Qwen3ASRCheckpoint:
    def __init__(self, backbone, projector):
        self.backbone, self.projector = GGUF(backbone), GGUF(projector)
        self.config = Qwen3ASRConfig.from_gguf(
            self.backbone.metadata, self.projector.metadata
        )
        if self.backbone.tensors.keys() & self.projector.tensors.keys():
            raise ValueError("duplicate tensors across the Qwen3 ASR GGUF pair")
        self.tensors = self.backbone.tensors | self.projector.tensors
        output = self.tensors.get("output.weight", self.tensors[TOKEN_EMBEDDING])
        if (
            output.shape != (self.config.vocab, self.config.width)
            or self.tensors[TOKEN_EMBEDDING].shape != output.shape
        ):
            raise ValueError("invalid Qwen3 ASR vocabulary projection")
        self.head = output.name

    def source(self, name):
        return self.backbone if name in self.backbone.tensors else self.projector

    def packed(self, name):
        source = self.source(name)
        if name != CONV_PROJECTION:
            return source.packed(name)
        # The checkpoint flattens channel then frequency. The convolution
        # produces frequency then channel so each spatial row is contiguous.
        channels = self.tensors["a.conv2d.3.weight"].shape[0]
        dtype = "float16" if self.tensors[name].type == 1 else "float32"
        if self.tensors[name].type not in (0, 1):
            raise ValueError("Qwen3 ASR convolution projection must be F16 or F32")
        weights = (
            source.packed(name)
            .view(dtype)
            .reshape(self.config.audio_width, channels, -1)
        )
        return (
            np.ascontiguousarray(weights.transpose(0, 2, 1)).view("uint8").reshape(-1)
        )

    def discard(self, name):
        self.source(name).discard(name)

    def buffer(
        self, graph, name, *, provider, half=False, tap_major=False, aligned=False
    ):
        from akas.quantization.weights.ggml.definition import weight_storage

        info = self.tensors[name]
        if half and info.type == 8:
            name += HALF_SUFFIX
        if tap_major:
            name += TAP_SUFFIX
        if aligned:
            name += ALIGNED_SUFFIX
        if name not in graph.buffers:
            size, dtype = weight_storage(
                1 if half else info.type,
                prod(info.shape),
                packed_words=provider == "webgpu",
            )
            if aligned:
                size = prod(info.shape) // 32 * 9
            graph.buffer(name, (size,), dtype, role="weight")
        return graph.buffers[name]

    def upload(self, device, names):
        from types import SimpleNamespace

        from akas.checkpoints.upload import UploadBatch, upload_packed
        from akas.quantization.weights.ggml.webgpu import q8_aligned_words

        with ExitStack() as cleanup:
            weights = upload_packed(
                device,
                SimpleNamespace(
                    tensors={
                        name: self.tensors[name]
                        for name in names
                        if name in self.tensors
                    },
                    packed=self.packed,
                    discard=self.discard,
                ),
            )
            for buffer in weights.values():
                cleanup.callback(buffer.release)
            with UploadBatch(device) as transfers:
                for name in names:
                    if name in weights:
                        continue
                    if name.endswith(ALIGNED_SUFFIX):
                        base = name.removesuffix(ALIGNED_SUFFIX)
                        host = q8_aligned_words(self.packed(base))
                        self.discard(base)
                    else:
                        base = name.removesuffix(HALF_SUFFIX).removesuffix(TAP_SUFFIX)
                        host = self.source(base).array(base, dtype=np.float16)
                        if name.endswith(TAP_SUFFIX):
                            host = np.ascontiguousarray(host.transpose(0, 2, 3, 1))
                    host = host.reshape(-1)
                    buffer = device.from_numpy(host)
                    transfers.add(host.nbytes)
                    weights[name] = buffer
                    cleanup.callback(buffer.release)
            cleanup.pop_all()
        return weights

    def inventory(self):
        return {
            name: dict(Counter(info.encoding for info in source.tensors.values()))
            for name, source in (
                ("backbone", self.backbone),
                ("projector", self.projector),
            )
        }
