"""Independent EOS must stop one slot without terminating sibling decoding."""
from types import SimpleNamespace
import numpy as np
from akas.generation.greedy import generate_batch


def test_slots_stop_independently_at_eos():
    masks = []
    model = SimpleNamespace(
        batch=3,
        capacity=16,
        _check=lambda: None,
        reset=lambda: None,
        prefill=lambda prompts, read: None,
        tokenizer=SimpleNamespace(eos=9, decode=lambda ids: str(ids)),
        buffers={"sampled": SimpleNamespace(to_numpy=lambda: np.array([9, 2, 3]))},
    )
    samples = iter([np.array([9, 9, 4]), np.array([9, 9, 9])])

    def decode(count, active):
        assert count == 1
        masks.append(active.copy().tolist())
        return next(samples)

    model.decode = decode
    results = generate_batch(model, [[1], [1, 2], [1]], max_tokens=8)
    assert [r["generated_tokens"] for r in results] == [[9], [2, 9], [3, 4, 9]]
    assert masks == [[0, 1, 1], [0, 0, 1]]
