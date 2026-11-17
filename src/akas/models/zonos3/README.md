# ZONOS3-Mini

`Zonos3MiniSynthesis` uses a Q8_0 backbone and F32 `keep2_plain` ZAC
projector. `config.py` parses and validates architecture geometry, and
`weights.py` loads and uploads the GGUF tensors. `model.py` declares the
graphs and runs synthesis through the shared `Forward` executor. The backbone
shares attention and Gated DeltaNet state across prefill and single-row phases.
The speech compilation profile builds the graphs and publishes `speech.json`.

The WebGPU path uses hardware Vulkan for normalized text in the 43 languages
listed in the [model card](https://huggingface.co/Zyphra/ZONOS3-Mini#languages),
with the projector's predefined voices. It returns mono 44.1 kHz audio. Compile frame
capacities for the expected output length. The runtime selects the smallest
compiled codec capacity that can hold the generated frames.

The `lang` setting selects a matching predefined voice when the caller omits
`voice`. The GGUF ships voices for 10 languages. Other languages use the
checkpoint's default voice unless the caller chooses a different preset.
`GET /props` exposes the language-to-voice map and the fallback.

The Python API also accepts `synthesize(text, lang=..., speaker_row=row)` for
an offline speaker encoder output. Pass the single F32 row of shape
`(1, config.width)` after the matching GGUF speaker projection. The projector
maps the raw speaker embedding to this row before synthesis.

```sh
WGPU_BACKEND_TYPE=Vulkan uv run --no-sync akas compile \
  --definition akas.models.zonos3:Zonos3MiniSynthesis --profile speech \
  --checkpoint build/checkpoints/zonos3-mini/ZONOS3-Mini-Q8_0.gguf \
  --checkpoint-options '{"projector":"build/checkpoints/zonos3-mini/mmproj-ZONOS3-Mini-F32.gguf"}' \
  --config '{"context":1024,"prefill_rows":64,"codec_frames":[32,64]}' \
  --out build/zonos3-mini/speech
```

Set `$env:WGPU_BACKEND_TYPE = 'Vulkan'` before the command in PowerShell. For longer prompts, a larger prefill capacity reduces repeated backbone dispatches and active-expert reads. The validated 256-row configuration uses `prefill_rows: 256`; it retains the same decode, sampling, codec, and waveform path and is within the established backbone reference tolerance.

Use the [speech benchmark commands](../../../../tools/baselines/README.md#speech-synthesis-demo)
to compare with llama.cpp Vulkan under the same request and audio protocol.
On the RX 6700 XT, the [final 43-language fixed-voice benchmark](https://github.com/kreasof-ai/akas/blob/b45b707fb78c9333ed37a49f860a8d6d2103c625/benchmarks/reports/20261117T121145Z-48e3bd2d83ae/COMPARISON.md)
completed 253/258 cases with **1.534× wall-time** and **1.712× audio-throughput**
speedup over llama.cpp Vulkan, and **1.876× wall-time speedup** over llama.cpp
CPU. The earlier 64-row configuration reached 1.369× and 1.477× on the same
sample. The generated report remains unqualified until the speech workload has
full-model reference qualification.
