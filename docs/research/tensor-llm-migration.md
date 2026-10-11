# Migration from Tensor LLM

The inference package, model producers, benchmark harness and model-specific tests
now belong to AKAS. Tensor retains its compiler, providers, runtime, artifact
format and generic kernel benchmarks. Historical research in Tensor remains
unchanged; retained model evidence is copied here for reference.

| Previous module | AKAS owner |
| --- | --- |
| `tensor_llm.common.plan` | `akas.execution.plan`, `akas.tensor.executor` |
| `tensor_llm.common.state` | `akas.memory.state`, `akas.ops.hybrid_state` |
| `tensor_llm.common.artifacts` | `akas.artifacts.kernels` |
| `tensor_llm.common.checkpoint` | `akas.checkpoints.safetensors` |
| `tensor_llm.common.gguf` | `akas.checkpoints.gguf` |
| `tensor_llm.common.scheduler` | `akas.scheduling.scheduler`, `akas.requests` |
| `tensor_llm.common.acceptance`, `lookup` | `akas.generation` |
| `tensor_llm.common.tokenizer` | `akas.text.tokenizer` |
| `tensor_llm.lfm2.model` | `akas.models.lfm2` definition; `akas.execution.packed`, generation and artifact components |
| `tensor_llm.qwen35` | `akas.models.qwen35` definition; shared block-FP8 execution and operations |
| `tensor_llm.qwen35.dense.engine` | `akas.models.qwen35` definition; `akas.execution.batched` |
| `tensor_llm.qwen35.dense.server` | `akas serve` with `examples.serving:qwen35_dense` |
| `tensor_llm.kernels` | `akas.kernels` operation families |
| `benchmarks.lfm2.producer` | `akas compile --profile packed` |
| `benchmarks.qwen35.*_producer` | `akas.compilation.profiles` and `akas.compilation.transforms` |
| Kernel/serving beam search runners | `benchmarks.qwen35.kernel_search`, `benchmarks.qwen35.serving_search` |

Public model entry points remain lazy exports from `akas`. The CLI becomes
`akas inspect` / `akas generate`; dense serving becomes `akas serve`.

The legacy inference document and research reports record historical commands,
package names, environment layouts and qualification results. They are preserved
as historical evidence, not the new setup guide. Use the root README for current
installation and commands. Regenerate implementation-bound bundles for AKAS.
