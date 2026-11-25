"""Compile and run the offline Inworld Voice Isolator Forward bundle."""

from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np

from akas.artifacts.read import read_forward_set
from akas.execution.forward import Forward
from akas.execution.workspace import CompactScratchPool
from akas.graph import Graph
from akas.memory.transfer import write
from akas.models.voice_isolator import _composition
from akas.models.voice_isolator.audio import float_wav, wav_samples
from akas.models.voice_isolator.weights import VoiceIsolatorCheckpoint


@dataclass(frozen=True)
class Enhancement:
    voice: np.ndarray
    background: np.ndarray
    sample_rate: int
    chunk_count: int
    timings: dict | None = None


def compose(checkpoint, *, phase, rows, provider="webgpu"):
    if provider != "webgpu" or phase != "enhance":
        raise ValueError("offline Voice Isolator requires a WebGPU enhance graph")
    return _composition.enhance(checkpoint, rows)


class VoiceIsolator:
    checkpoint_type = VoiceIsolatorCheckpoint
    forward_index = ("speech.json", "akas.voice-isolation-program.v1")
    compose = staticmethod(compose)

    @classmethod
    def forward_phases(cls, checkpoint, *, provider, chunks=(144000, 384000)):
        if provider != "webgpu":
            raise ValueError("offline Voice Isolator requires WebGPU")
        if not chunks or any(
            type(samples) is not int
            or not 1 <= samples <= checkpoint.config.chunk_samples
            for samples in chunks
        ):
            raise ValueError("invalid Voice Isolator compiled audio capacities")
        return dict(
            sample_rate=checkpoint.config.sample_rate,
            weight_dtype=checkpoint.weight_dtype,
            fp32_projections=list(checkpoint.fp32_projections),
            chunks=sorted(set(chunks)),
        ), (
            (
                "enhance",
                samples,
                cls.compose(
                    checkpoint, phase="enhance", rows=samples, provider=provider
                ),
            )
            for samples in sorted(set(chunks))
        )

    def __init__(self, checkpoint, directory, device, *, compact_scratch=True):
        self.checkpoint, self.config, self.device = (
            checkpoint,
            checkpoint.config,
            device,
        )
        self._resources = ExitStack()
        filename, schema = self.forward_index
        phases = read_forward_set(
            directory, filename, schema=schema, provider=device.info["provider"]
        )
        definitions = phases["enhance"]
        self.capacities = sorted(map(int, definitions))
        shared = {}
        for manifest in definitions.values():
            for spec in manifest["forward"]["buffers"]:
                if spec["role"] != "weight" or spec.get("alias_of"):
                    continue
                if shared.setdefault(spec["name"], spec) != spec:
                    raise ValueError(
                        f"incompatible Voice Isolator weight: {spec['name']}"
                    )
        try:
            supplied = checkpoint.upload(device, sorted(shared))
            for buffer in supplied.values():
                self._resources.callback(buffer.release)
            self._weight_bytes = sum(buffer.nbytes for buffer in supplied.values())
            self._scratch = None
            if compact_scratch and "capacity_prefix_views" in device.capabilities:
                self._scratch = CompactScratchPool(
                    device,
                    (
                        Graph.from_metadata(phase["forward"])
                        for phase in definitions.values()
                    ),
                )
                self._resources.callback(self._scratch.close)
            self._forwards = {}
            for samples, manifest in definitions.items():
                names = {
                    spec["name"]
                    for spec in manifest["forward"]["buffers"]
                    if spec["role"] == "weight" and not spec.get("alias_of")
                }
                forward = Forward(
                    directory,
                    device,
                    {name: supplied[name] for name in names},
                    manifest=manifest,
                    graphs=True,
                    scratch_pool=self._scratch,
                    allocate_io=True,
                )
                self._forwards[int(samples)] = self._resources.enter_context(forward)
        except BaseException:
            self.close()
            raise

    @property
    def metadata(self):
        return dict(
            device=dict(self.device.info, name=self.device.info["adapter"]["device"]),
            sample_rate_hz=self.config.sample_rate,
            weight_dtype=self.checkpoint.weight_dtype,
            projection_activation_dtype=self.checkpoint.weight_dtype,
            fp32_projections=list(self.checkpoint.fp32_projections),
            buffer_bytes=dict(
                weights_and_constants=self._weight_bytes,
                shared_scratch=self._scratch.nbytes if self._scratch else 0,
                workspaces={
                    str(samples): (
                        sum(
                            forward.buffers[name].nbytes
                            for name, spec in forward.definition.buffers.items()
                            if spec.role in ("input", "output")
                            and spec.alias_of is None
                        )
                        if self._scratch
                        else sum(buffer.nbytes for buffer in forward.owned)
                    )
                    for samples, forward in self._forwards.items()
                },
            ),
            compiled_samples=self.capacities,
            fft_size=self.config.n_fft,
            hop_length=self.config.hop_length,
            bands=len(self.config.edges),
            width=self.config.width,
        )

    def enhance_chunk(self, samples):
        values = np.ascontiguousarray(samples, dtype="float32")
        if values.ndim != 1 or not 1 <= len(values) <= self.config.chunk_samples:
            raise ValueError("one enhancement chunk needs 1 to 8 seconds of mono audio")
        if not np.isfinite(values).all():
            raise ValueError("audio contains NaN or infinite samples")
        if len(values) not in self._forwards:
            raise ValueError(
                "this exact audio length has no compiled Voice Isolator graph"
            )
        forward = self._forwards[len(values)]
        write(self.device, forward.buffers["audio"], values)
        enhanced = forward.graph.launch(readback=forward.buffers["enhanced_audio"])
        return enhanced[: len(values)]

    def isolate(self, audio, *, profile=False):
        samples, rate = (
            wav_samples(audio)
            if isinstance(audio, (bytes, bytearray))
            else (np.ascontiguousarray(audio, dtype="float32"), self.config.sample_rate)
        )
        if rate != self.config.sample_rate or samples.ndim != 1 or not len(samples):
            raise ValueError("requires nonempty mono 48 kHz audio")
        if not np.isfinite(samples).all():
            raise ValueError("audio contains NaN or infinite samples")
        chunk, overlap = (
            self.config.chunk_samples,
            int(self.config.overlap_seconds * rate),
        )
        if len(samples) <= chunk:
            voice = self.enhance_chunk(samples)
            return Enhancement(voice, samples - voice, rate, 1, None)
        from time import perf_counter

        timings = dict(prepare=0.0, gpu=0.0, combine=0.0) if profile else None
        start = perf_counter()
        starts = list(range(0, len(samples) - chunk, chunk - overlap)) + [
            len(samples) - chunk
        ]
        out = np.zeros_like(samples)
        weight = np.zeros_like(samples)
        for index, first in enumerate(starts):
            end = first + chunk
            window = np.ones(chunk, dtype="float32")
            if index > 0:
                fade_in = starts[index - 1] + chunk - overlap - first
                window[:fade_in] = 0
                window[fade_in : fade_in + overlap] = np.linspace(
                    0, 1, overlap, dtype="float32"
                )
            if end < len(samples):
                window[-overlap:] = np.linspace(1, 0, overlap, dtype="float32")
            if timings is not None:
                timings["prepare"] += perf_counter() - start
                start = perf_counter()
            enhanced = self.enhance_chunk(samples[first:end])
            if timings is not None:
                timings["gpu"] += perf_counter() - start
                start = perf_counter()
            out[first:end] += enhanced * window
            weight[first:end] += window
            if timings is not None:
                timings["combine"] += perf_counter() - start
                start = perf_counter()
        voice = out / np.maximum(weight, 1e-6)
        return Enhancement(voice, samples - voice, rate, len(starts), timings)

    def close(self):
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


__all__ = ["Enhancement", "VoiceIsolator", "compose", "float_wav"]
