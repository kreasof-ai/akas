# Inworld Voice Isolator

`VoiceIsolator` runs the released **offline** `inworld/voice-isolator`
checkpoint as a compiled WebGPU Forward bundle. It processes mono 48 kHz audio
with the exact STFT, 46-band dual-path network, complex correction and iSTFT.
No learned computation remains on the host during a chunk.

Use the released FP32 checkpoint (691 tensors, 64,047,112 parameters).
The realtime checkpoint and streaming API are outside this offline demo.
The optional W16A16 projection path saves weight memory but did not pass the
11-second golden waveform check, even with selected projections restored to
FP32. It is experimental and not the recommended configuration.

## Compile the FP32 bundle

Place the checkpoint at the path in the
[benchmark model declaration](../../../../benchmarks/models/voice-isolator-v1.json).

```powershell
$env:WGPU_BACKEND_TYPE = 'Vulkan'
uv run --no-sync akas compile --profile speech `
  --definition akas.models.voice_isolator:VoiceIsolator `
  --checkpoint build/checkpoints/voice-isolator-v1/denoiser.safetensors `
  --out build/voice-isolator/f32
```

Each capacity is the exact sample count of one full network pass. Short clips
are not padded into an eight-second forward pass: non-causal attention would
otherwise see the padding. Long audio uses complete eight-second windows, a
six-second advance, and the upstream two-second linear crossfade. The last
window ends at the recording's end.

For Python callers, `akas.models.voice_isolator.interface.service(checkpoint,
bundle)` opens the compiled model as a context manager. Call
`model.isolate(wav_bytes)` within that context to get voice and background
samples. The enhancement benchmark uses the same factory.

## Shared GPU scratch

With Tensor's WebGPU `capacity_prefix_views` capability, the two exact-length
graphs share a pool of typed scratch allocations. Non-overlapping
intermediates can reuse a larger buffer even when shapes differ. Weights and
audio input/output buffers remain separate. Older Tensor runtimes fall back
to separate scratch per graph. Check `buffer_bytes.shared_scratch` in the
native run metadata to confirm the pool is active.

To use a local Tensor checkout that contains WebGPU prefix-view support:

```powershell
$env:PYTHONPATH = "$((Resolve-Path '../tensor/src').Path)"
uv run --no-sync python -c "from tensor.providers.webgpu.device import Device; assert 'capacity_prefix_views' in Device.capabilities"
```

See the [benchmark workflow](../../../../tools/baselines/README.md#voice-enhancement-demo)
for the short CPU reference, the full GPU workload, and generated report.
