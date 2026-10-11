"""Greedy prompt, termination and result policy for independent requests."""

from __future__ import annotations
import numpy as np
from akas.execution.independent.greedy import advance_greedy


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
