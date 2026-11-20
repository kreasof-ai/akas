"""Architecture and fixed-shape geometry of the offline Voice Isolator."""

from dataclasses import dataclass


DEFAULT_BAND_PLAN = (
    (100, 1000),
    (250, 4000),
    (500, 8000),
    (1000, 16000),
    (2000, 24000),
)


@dataclass(frozen=True)
class VoiceIsolatorConfig:
    width: int
    block_types: tuple[str, ...]
    num_heads: int
    n_fft: int
    hop_length: int
    sample_rate: int
    band_plan: tuple[tuple[int, int], ...]
    correction_bound: float
    chunk_seconds: float
    overlap_seconds: float
    exclusive_self_attention: bool
    exclusive_axes: str
    mamba_state: int
    mamba_conv: int
    mamba_expand: int
    causal: bool
    lookahead_frames: int
    lookahead_frames_choices: tuple[int, ...]
    dual_mode: bool
    mode_film: bool
    attention_window_frames: int
    offline_passes: int

    @classmethod
    def from_metadata(cls, raw):
        required = {
            "width": 512,
            "block_types": [
                "transformer",
                "mamba",
                "transformer",
                "mamba",
                "transformer",
                "mamba",
                "transformer",
            ],
            "num_heads": 8,
            "n_fft": 2048,
            "hop_length": 480,
            "sample_rate": 48000,
            "band_plan": [list(pair) for pair in DEFAULT_BAND_PLAN],
            "correction_bound": 0.5,
            "chunk_seconds": 8,
            "overlap_seconds": 2,
            "exclusive_self_attention": True,
            "exclusive_axes": "time",
            "mamba_state": 16,
            "mamba_conv": 4,
            "mamba_expand": 2,
            "causal": False,
            "lookahead_frames": 0,
            "lookahead_frames_choices": [],
            "dual_mode": False,
            "mode_film": False,
            "attention_window_frames": 0,
            "offline_passes": 1,
        }
        ignored = {
            "activation_checkpointing",
            "checkpoint_every",
            "attention_backend",
            "offline_mode_prob",
            "dual_distill",
            "hint_sha256",
            "hint_bf16",
            "grid_width",
            "grid_hidden",
            "grid_kernel",
            "grid_stride",
            "grid_heads",
            "grid_qk_channels",
            "grid_output",
            "grid_normalize_input",
            "decision_head",
            "representation",
            "decoder_type",
            "hint_checkpoint",
        }
        unsupported = set(raw) - set(required) - ignored
        if unsupported or raw.get("representation", "bands") != "bands":
            raise ValueError("unsupported Voice Isolator checkpoint architecture")
        for key, expected in required.items():
            if raw.get(key, expected) != expected:
                raise ValueError(f"unsupported Voice Isolator setting: {key}")
        return cls(
            width=512,
            block_types=tuple(required["block_types"]),
            num_heads=8,
            n_fft=2048,
            hop_length=480,
            sample_rate=48000,
            band_plan=DEFAULT_BAND_PLAN,
            correction_bound=0.5,
            chunk_seconds=8.0,
            overlap_seconds=2.0,
            exclusive_self_attention=True,
            exclusive_axes="time",
            mamba_state=16,
            mamba_conv=4,
            mamba_expand=2,
            causal=False,
            lookahead_frames=0,
            lookahead_frames_choices=(),
            dual_mode=False,
            mode_film=False,
            attention_window_frames=0,
            offline_passes=1,
        )

    @property
    def bins(self):
        return self.n_fft // 2 + 1

    @property
    def head_dim(self):
        return self.width // self.num_heads

    @property
    def inner(self):
        return self.width * self.mamba_expand

    @property
    def dt_rank(self):
        return (self.width + 15) // 16

    @property
    def chunk_samples(self):
        return int(self.chunk_seconds * self.sample_rate)

    @property
    def edges(self):
        edges, start = [], 0
        hz_per_bin = self.sample_rate / self.n_fft
        for width_hz, upto_hz in self.band_plan:
            width = max(1, round(width_hz / hz_per_bin))
            upto = min(self.bins, round(upto_hz / hz_per_bin))
            while start < upto:
                end = min(start + width, upto)
                edges.append((start, end))
                start = end
        if start < self.bins:
            edges.append((start, self.bins))
        return tuple(edges)

    def frames(self, samples):
        if samples < 1:
            raise ValueError("audio must contain at least one sample")
        return samples // self.hop_length + 1
