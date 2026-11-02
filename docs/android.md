# Experimental Android execution

The Android prototype runs compiled AKAS WebGPU programs on the phone through
wgpu-native/Vulkan. Compilation runs on the desktop. The consumer uses an
ARM64 CPython runtime and Tensor's ARM64 native WebGPU plan helper.

The tested device is Xiaomi 21081111RG, Android 14, MT6893, Mali-G77 MC9.
Its Vulkan compute subgroups contain 16 lanes. Circular attention now has a
shared-memory reduction for subgroups smaller than 32, and Q6_K decode can
assign its 256 coefficients across 16 lanes.

The completed GPU smoke cases are prefixes 32, 512 and 2048 with 32 output
tokens each. The 8192-token GPU attempt ended in a phone reboot with Android
boot reason `hang_detect`; the cause is unresolved. The CPU comparison
completed all four prefixes. The GPU command below uses the completed range.

## Compile an LFM3 consumer program

Use the Tensor revision pinned by this checkout. No GPU search is performed by
this command; it selects an explicit experimental schedule. It leaves the
general WebGPU defaults unchanged.

```powershell
python -m tools.android_compile_lfm3 `
  --checkpoint build/checkpoints/lfm3-1.7b/q4_0.gguf `
  --out build/android/mali-program `
  --runtime-out build/android/mali-runtime `
  --stage-weights
```

This compiles B1, chunks 1/32/128, context capacity 8320, FP16 KV, packed
GGML weights, short FP16 prefill accumulations with F32 totals, and 16-lane
decode schedules. `--stage-weights` uses temporary FP16 projection operands
for chunks of at least 128; the persistent checkpoint weights stay packed.
`--half-layout`, `--half-k`, `--decode-layout` and `--chunk` select alternatives.
The runtime directory contains the referenced JSON and kernel artifacts.

## Build the CPU comparison

The comparison uses a small C API runner rather than a server. The tested
llama.cpp revision is `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552` (b11946).
Check out that revision under `build/android/llama-src` before configuring.
The tested toolchain is Android NDK r27d, Clang 18.0.4.

```powershell
cmake -S tools/android -B build/android/llama-build -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$env:ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" `
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLAMA_SOURCE="$PWD/build/android/llama-src" `
  -DGGML_CPU_KLEIDIAI=ON `
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 `
  -DGGML_CCACHE=OFF -DLLAMA_BUILD_NUMBER=11946 `
  -DLLAMA_BUILD_COMMIT=8ca4060b7e30e76baaa5670ee1ec1b760b7f8552
cmake --build build/android/llama-build --parallel 8

python -m tools.android_llama_benchmark --adb "$env:LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe" `
  --serial DEVICE_SERIAL --threads 4 --batch-threads 8 `
  --decode-mask f0 --batch-mask ff --generated 32 `
  --prefixes 32 512 2048 8192 --out build/android/llama-cpu.json
```

The masks above are specific to this device: decode uses its four big cores,
and prefill uses all eight cores. The runner creates separate thread pools,
sets GPU layers/offload to zero, uses FP16 KV and flash attention, and clears
both KV and recurrent state before each request. It verifies the phone's
checkpoint hash before running.

## Run AKAS on the phone

The development device already has the consumer runtime under
`/data/local/tmp/akas/runtime` and the checkpoint at
`/data/local/tmp/akas/lfm3-qad-q4_0.gguf`. That runtime was assembled for testing;
it is not a packaged Android application. Its tested components are CPython
3.12.15, NumPy 1.26.2, cffi 1.17.1, wgpu-py 0.29.0, wgpu-native v27.0.2.0,
and the checkout's AKAS/Tensor sources and native plan helper.

```powershell
adb -s DEVICE_SERIAL push build/android/mali-runtime /data/local/tmp/akas/mali-runtime
adb -s DEVICE_SERIAL push tools/android_lfm3_benchmark.py /data/local/tmp/akas/benchmark.py
adb -s DEVICE_SERIAL shell 'cd /data/local/tmp/akas && LD_LIBRARY_PATH=/data/local/tmp/akas/runtime/lib PYTHONHOME=/data/local/tmp/akas/runtime WGPU_LIB_PATH=/data/local/tmp/akas/runtime/lib/libwgpu_native.so WGPU_BACKEND_TYPE=Vulkan XDG_RUNTIME_DIR=/data/local/tmp/akas OPENBLAS_NUM_THREADS=1 runtime/bin/python -u benchmark.py --checkpoint lfm3-qad-q4_0.gguf --program mali-runtime --prefixes 32 512 2048 --generated 32 --out result.json'
adb -s DEVICE_SERIAL pull /data/local/tmp/akas/result.json build/android/result.json
```

Both runners use the same deterministic synthetic token IDs, seed
`17 + 4099 + prefix`, with no prefix reuse and no early EOS. They warm up once,
then retain a single measured sample for each prefix. Loading and pipeline
creation are excluded. Prefill includes the first greedy prediction; decode
measures the remaining 31 steps, giving 32 output tokens in total. AKAS includes
its on-device Python controls and token readbacks. The report captures every
output ID and marks that this smoke test is not an independent model-quality
qualification. `--demo` also runs a real-text question.

The scripts retain first-prefill logits outside the timed sections:
`result-prefixN-logits.npy` for AKAS, and `llama-inputs/logits-N.f32` for
llama.cpp. Pull these explicitly to compare numerical behavior. Synthetic
prompts can produce repetitive output; token agreement alone does not measure
language-model quality.

## Check the portable kernels

```powershell
$env:AKAS_LFM3_WEBGPU = '1'
$env:WGPU_BACKEND_TYPE = 'Vulkan'
python -m pytest tests/test_attention_small_subgroups.py tests/test_mobile_half_tiles.py tests/test_q6_small_subgroups.py -q
```

The attention tests force the fallback in the test shader on desktop GPUs and
compare it with an independent NumPy oracle. Projection tests exercise partial
tiles, paired FFN outputs, and both scalar and vector 16-lane Q6 reductions.
`tools/android_kernel_profile.py` is a rough per-call latency diagnostic: its
synchronization overhead means its sums are not whole-plan GPU time.
