"""LFM2 greedy generation using its request workspaces and captured plans."""

from __future__ import annotations
import numpy as np


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
            if self.provider == "webgpu":
                next_token = self.greedy[1].launch(
                    readback=self.workspaces[1]["tokens"]
                )
            else:
                long = (
                    bool(self.long_plan) and self.position + 1 >= self.grouped_threshold
                )
                if self.graphs_enabled:
                    (self.long_greedy_graph if long else self.greedy[1]).launch()
                else:
                    self._submit(
                        self.long_greedy_plan if long else self.greedy_plans[1]
                    )
                next_token = self.workspaces[1]["tokens"].to_numpy()
            self.position += 1
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
