"""Research preset using one compiled program and the shared performance harness.

Legacy separate bundle/installer flags have been replaced by --policy and
--model-options. --workload retains actual request replay; --windows qualifies
multiple speculative geometries in one program. --prepared reuses its manifest.
"""

from akas.benchmarking.research import main as research_main

DEFINITION = "akas.models.qwen35.model:Qwen35Batch"
PRESET = {'batches': [8, 16, 32, 64], 'batch': 8, 'chunk': 512, 'windows': [32], 'context': 48000, 'target': 'sm_90a', 'prompt': 32, 'generated': 32, 'model_options': {'kv_dtype': 'fp8'}, 'policy': {'moe': {'compact': True, 'hopper': True, 'bf16_mma': True, 'mma_reduction': 32, 'mma_reorder': True}, 'attention': {'splits': 16, 'hopper': True, 'key_rows': 32, 'query_tokens': 8}}}


def main():
    research_main(model_definition=DEFINITION, preset=PRESET, description=__doc__)


if __name__ == "__main__":
    main()
