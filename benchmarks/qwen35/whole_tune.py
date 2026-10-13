"""Research preset using one compiled program and the shared performance harness.

Legacy separate bundle/installer flags have been replaced by --policy and
--model-options. --workload retains actual request replay; --windows qualifies
multiple speculative geometries in one program. --prepared reuses its manifest.
"""

from akas.benchmarking.research import main as research_main

DEFINITION = "akas.models.qwen35.model:Qwen35Batch"
PRESET = {'batches': [1, 4, 16, 64, 256], 'batch': 1, 'chunk': 128, 'windows': [], 'context': 4096, 'target': None, 'prompt': 32, 'generated': 32, 'model_options': {'kv_dtype': 'fp8'}, 'policy': {}}


def main():
    research_main(model_definition=DEFINITION, preset=PRESET, description=__doc__)


if __name__ == "__main__":
    main()
