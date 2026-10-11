# Shared operation composition

These components implement operation behavior used by model plans. They do not
know checkpoint names, layer order, request admission policy or model families.

| Component | Behavior | Current callers |
| `decoder.DecoderBlock` | Compose normalization, mixer and feed-forward operations using declared weight roles | Both model families |
| `attention` / `gdn` / `shortconv` | Compose the selected mixer while preserving cache and recurrent updates | Packed, FP8 and BF16 contracts |
| `ffn` | Compose packed SwiGLU, routed/shared experts, or dense fused projections | All decoder contracts |
| --- | --- | --- |
| `linear.linear` | Derive projection geometry; bind BF16 or block-FP8 weights, with optional prequantized activations | MoE decode and prefill |
| `linear.packed_linear` | Compose a packed projection with an explicit quantization/profile and input preparation | LFM2 |
| `linear.split_linear` | Compose split/merge projection calls and borrow the output for singleton splits | Dense Qwen |
| `normalization.rms_norm` | Compose RMS or fused residual normalization, preserving kernel input names and diagnostic labels | LFM2, MoE decode/prefill, dense Qwen |
| `quantization.ActivationCache` | Reuse quantized scratch across readers; invalidate aliases when the source is overwritten | MoE prefill |
| `hybrid_state.allocate_hybrid_states` | Derive KV, recurrent and convolution storage from explicit geometry | All model implementations |

`execution.plan.KernelPlan` resolves a kernel specialization, applies an optional
numerical/profile parameter transform, binds positional or named arguments and
records calls in order. A cache can observe output writes during composition.
It borrows kernels and buffers; it does not destroy model resources.

Model plans map checkpoint tensors to these operations and compose the layer
sequence. Numerical variants remain explicit: native FP8 decode and prequantized
FP8 prefill have different argument contracts; packed LFM2 uses its recorded
quantization type; dense projections preserve split/merge arithmetic. Sharing
composition does not silently change precision, layouts or rounding.

Factories in `akas.kernels` implement the arithmetic. Tensor compiles those
factories and provides artifact execution. Add reusable operation composition
here when a new model can use the same contract; state commit and repair orchestration
belong to execution and generation components with the matching contract.
