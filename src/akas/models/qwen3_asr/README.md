# Qwen3 ASR

`Qwen3ASR` runs the [AtomicChat Rumble-2-Flash GGUF](https://huggingface.co/AtomicChat/Rumble-2-Flash-GGUF).
The demo uses the Q8_0 decoder and Q8_0 audio projector pinned in the
[benchmark model declaration](../../../../benchmarks/models/qwen3-asr.json).
Rumble-2-Flash derives from Qwen3-ASR-0.6B and has a 14-layer decoder and
18-layer audio encoder.

The runtime reads PCM16 WAV, mixes to mono and resamples to 16 kHz. It reuses
the sinc coefficients for each sample-rate phase and multiplies only the
nonzero log-Mel filter coefficients. Prepared WebGPU plans run the
convolutions, audio encoder, decoder and greedy vocabulary selection over
hardware Vulkan.

The checkpoint's Q8_0 coefficients and scales are repacked into aligned words
without requantizing the weights. Encoder and decoder projections use
block-32 signed-byte activations and FP32 block accumulation. The frontend
uses the checkpoint's F16 convolution weights. For the two 480-channel
convolutions, bias and GELU write an F16 feature map. Each GPU workgroup
reads its convolution taps from that map into a shared tile, eliminating
the expanded patch matrices. The final convolution chunk remains padded to 100 frames for parity
with llama.cpp b12521. Audio attention covers up to eight chunks per window.
The model card documents possible extra words on short clips with this
padding behavior.

Place the two GGUF files at the paths in the benchmark model declaration.
Compile the speech bundle with the existing forward-set profile.

```sh
WGPU_BACKEND_TYPE=Vulkan uv run --no-sync akas compile \
  --definition akas.models.qwen3_asr:Qwen3ASR --profile speech \
  --checkpoint build/checkpoints/qwen3-asr/Rumble-2-Flash-Q8_0.gguf \
  --checkpoint-options '{"projector":"build/checkpoints/qwen3-asr/mmproj-Rumble-2-Flash-Q8_0.gguf"}' \
  --out build/qwen3-asr/speech
```

On PowerShell, set `$env:WGPU_BACKEND_TYPE = 'Vulkan'` before running the
command. The default context is 512 tokens. The runtime selects from audio
encoder plans with 28, 52, 68, 80 or 104 rows and decoder prefill plans with
32, 64, 80 or 96 rows. Each convolution chunk writes its features into the
encoder input on the GPU. Only the last request prefill chunk returns hidden
states to the host. At the default context, the 96-row prefill plan
splits attention over four key ranges.
It prepares the model's fixed system and audio-start prefix in KV at startup.
The audio embeddings, prompt and output token limit must fit the compiled
context. After the first greedy prediction, prepared decode blocks keep the
token and position on the GPU for up to four steps. A device journal collects
new token IDs and stops collecting at either end marker. The host reads the
journal once per block. Older bundles continue to use single-step readback.
With the default context, prompts of at least 128 tokens use sixteen GPU
workgroups per attention head to scan the KV cache in parallel. The decoder
merges their partial softmax results on the GPU. Shorter prompts keep the
single-partition decode path.

Serve the compiled bundle with the transcription interface.

```sh
WGPU_BACKEND_TYPE=Vulkan uv run --no-sync akas serve --speech \
  --factory akas.models.qwen3_asr.interface:service \
  --config '{"checkpoint":"build/checkpoints/qwen3-asr/Rumble-2-Flash-Q8_0.gguf","projector":"build/checkpoints/qwen3-asr/mmproj-Rumble-2-Flash-Q8_0.gguf","bundle":"build/qwen3-asr/speech"}'
```

`POST /v1/chat/completions` accepts a single WAV audio input with
`chat_template_kwargs.language`. `GET /props` lists the supported language
codes, input sample rate and compiled context.

See the [transcription benchmark](../../../../tools/baselines/README.md#speech-transcription-demo)
for the matched request, frozen baseline and report commands. Set
`AKAS_ASR_PROFILE=1` for an AKAS run to retain per-request phase timings in its
record, including CPU features, convolution, audio transformer, prefill, head
and decode.
