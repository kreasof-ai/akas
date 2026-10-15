# Kernel infrastructure

This package supports every operation and codec. It is separate from the uniform
six-file operation-family tree.

| Module | Responsibility |
| --- | --- |
| [kernel.py](kernel.py) | `Tensor`, `output`, `@kernel` and the single typed-function-to-compiler boundary. |
| [definition.py](definition.py) | `KernelSpec`, `Capabilities`, schedule validation and `define`. |
| [contracts.py](contracts.py) | Implementation declarations, output ABIs and independent qualification types. |
| [registry.py](registry.py) | Discover operation/codec `definition.py` files, resolve implementations and graph rules. |
| [reductions.py](reductions.py) | Shared-memory tree reduction helper. |
| [webgpu.py](webgpu.py) | Subgroup and segmented reduction primitives. |

## Inline primitives

A reduction macro is a building block inside a kernel, not another operation or
launch. Share the tree algorithm where its contract fits; specialize warp or
subgroup instructions where necessary. CUDA warp shuffles also appear in
accelerated attention/projection bodies. An inline helper's directory is not a
backend support matrix.

The old state mapping utility is a compiler IR transformation and lives in
[compilation/transforms/state.py](../compilation/transforms/state.py).

## Adding an operation

Use the [operation guide](../ops/README.md) and a family package. Do not add a new
mathematical operation or checkpoint format to the shared authoring/registry
infrastructure. Compiler imports remain lazy, so operation discovery and artifact
inspection work without loading TileLang or a device compiler.
