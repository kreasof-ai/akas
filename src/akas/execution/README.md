# Shared execution

Read `executor.py` first. It owns request admission/release, occupied slots,
positions, pending verification, batch validation, prefill/decode dispatch,
owner-thread checks and shutdown. A slot identifies persistent request state;
it does not specify a GPU row.

Numerical bindings implement four hooks: reset admitted state, release state,
execute a batch and close owned resources. They share the executor lifecycle.

| Binding | Forward layout | State/precision constraints |
| --- | --- | --- |
| `bindings/packed/` | Serialized independent handles sharing weights and scratch | Packed GGUF projections; existing FP16 KV and FP32 convolution |
| `bindings/block_fp8/` | Stable GPU rows with explicit inactive masks | Block-FP8 weights and qualified fixed kernel geometry |
| `bindings/bf16/` | Row-to-request mapping over a persistent state pool | BF16 projections and the existing KV/recurrent layouts |

`plan.py` owns kernel resolution and named/ordered argument binding. Provider
submission and capture use `akas.tensor.executor`; numerical kernels stay in
`akas.ops`. Bindings retain specialized prefill, draft, verification and repair
plans where their buffer ABIs or commit semantics differ.

The public LFM2 default-request API and Qwen direct-forward helpers are retained.
Scheduled execution uses the shared executor, including the generic handle binding.
The compiler remains optional. This merge does not change compiled batch/chunk
geometries or introduce dynamic kernels.
