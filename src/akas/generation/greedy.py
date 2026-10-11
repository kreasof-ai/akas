"""Greedy prompt, termination and result policy for independent requests."""

from __future__ import annotations
import numpy as np
from akas.execution.bindings.packed.greedy import advance_greedy


def generate_greedy(self, prompt, *, max_tokens=128, chat=True, gpu_greedy=True):
    self._check_open()
    if type(max_tokens) != int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    tokens = self.tokenizer.chat(prompt) if chat else self.tokenizer.encode(prompt)
    if len(tokens) + max_tokens > self.context:
        raise ValueError("generation exceeds context capacity")
    if (self.provider == "webgpu" or self.cuda_optimized) and gpu_greedy:
        self.reset()
        next_token = self.forward(tokens, _greedy=True)
        generated = []
        for step in range(max_tokens):
            token = int(next_token[0])
            generated.append(token)
            if token == self.tokenizer.eos:
                break
            if step + 1 == max_tokens:
                break
            next_token = advance_greedy(self)
        return {
            "prompt_tokens": tokens,
            "generated_tokens": generated,
            "text": self.tokenizer.decode(generated),
        }
    self.reset()
    logits = self.forward(tokens)
    generated = []
    for _ in range(max_tokens):
        token = int(np.argmax(logits))
        generated.append(token)
        if token == self.tokenizer.eos:
            break
        logits = self.forward([token])
    return {
        "prompt_tokens": tokens,
        "generated_tokens": generated,
        "text": self.tokenizer.decode(generated),
    }


def generate_batch(model, prompts, *, max_tokens=128, chat=True):
    """Greedy generation over any batched token-step session.

    Each slot keeps its own sample and stops independently at EOS. Network and
    scheduling policies are left to callers; this loop also accepts token lists.
    """
    model._check()
    if type(max_tokens) is not int or max_tokens < 1 or len(prompts) != model.batch:
        raise ValueError("positive max_tokens and one prompt per slot required")
    encoded = []
    for prompt in prompts:
        if isinstance(prompt, str):
            if model.tokenizer is None:
                raise ValueError("text generation requires a tokenizer")
            prompt = (
                model.tokenizer.chat(prompt) if chat else model.tokenizer.encode(prompt)
            )
        encoded.append(list(prompt))
    if any(len(p) + max_tokens - 1 > model.capacity for p in encoded):
        raise ValueError("generation exceeds context capacity")
    model.reset()
    model.prefill(encoded, read=False)
    sampled = model.buffers["sampled"].to_numpy()
    generated = [[] for _ in prompts]
    active = np.ones(model.batch, dtype="int32")
    eos = model.tokenizer.eos if model.tokenizer is not None else None
    for step in range(max_tokens):
        for slot in range(model.batch):
            if active[slot]:
                token = int(sampled[slot])
                generated[slot].append(token)
                if token == eos:
                    active[slot] = 0
        if not active.any() or step + 1 == max_tokens:
            break
        sampled = model.decode(1, active=active)
    return [
        dict(
            prompt_tokens=p,
            generated_tokens=g,
            text=model.tokenizer.decode(g) if model.tokenizer is not None else None,
        )
        for p, g in zip(encoded, generated)
    ]
