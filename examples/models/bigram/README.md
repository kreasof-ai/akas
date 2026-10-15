# A small complete model

This deliberately simple model chooses the next token by looking up one row of
scores. It demonstrates the same `Definition` / `compose` contract as LFM and
Qwen, without adding a kernel, compiler, scheduler, or executor.

- `config.py` describes the checkpoint geometry.
- `weights.py` reads `config.json` and the F32 `scores[vocab, vocab]` tensor in
  `model.safetensors`, then uploads the declared storage.
- `model.py` exposes `compose` and declares the checkpoint reader.
- `__init__.py` owns the package's public model name.

Compile with `--definition examples.models.bigram:Bigram` and a configuration such
as `{"geometries":[[4,1]],"max_requests":4,"max_context":32}`. CUDA and WebGPU
use the same graph and their registered operation implementations. The example
supports chunk size 1 and token-ID requests; it does not contain a tokenizer.

```python
from akas import Engine
from examples.models.bigram import Bigram

with Engine.from_checkpoint(
    Bigram, checkpoint_directory, bundle_directory, device
) as engine:
    request = engine.submit([1, 2], max_tokens=4)
    while engine.step():
        pass
    print(request.output)
```
