"""Greedy prompt, termination and result policy for independent requests."""

from __future__ import annotations
import numpy as np


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


def generate_scheduled(model, prompt, *, tokenizer, max_tokens=128, chat=True, sampling="device"):
    """Generate a sequence through any scheduled executor without disturbing peers."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if sampling not in ("device", "host"):
        raise ValueError("sampling must be device or host")
    if tokenizer is None:
        raise ValueError("text generation requires a tokenizer")
    chunk = model.prefill_geometry() if hasattr(model,"prefill_geometry") else 32
    host_frames = {}
    if sampling == "host":
        if not hasattr(model,"batches") or not hasattr(model.batches,"select"):
            raise ValueError("host sampling requires declared logits features")
        geometries = {row["chunk"] for row in getattr(model.batches,"records",())
                      if row["phase"] in ("decode","prefill") and row["chunk"] <= chunk}
        for geometry in geometries or {chunk,1}:
            frame = model.batches.select(1,geometry)
            if "logits" not in frame.features:
                raise ValueError("host sampling requires declared logits features")
            name = frame.features["logits"]
            spec = frame.forward.definition.buffers[name]
            if spec.role != "output":
                raise ValueError("host sampling requires owned output logits")
            from math import prod

            if prod(spec.shape) != frame.rows * model.config.vocab:
                raise ValueError("host sampling requires one vocabulary row per request")
            host_frames[geometry] = frame

    def sample(predicted, geometry):
        if sampling == "device":
            return predicted[0]
        if geometry not in host_frames:
            frame = model.batches.select(1,geometry)
            if "logits" not in frame.features:
                raise ValueError("host sampling requires declared logits features")
            host_frames[geometry] = frame
        frame = getattr(model,"last_prefill_frame",None) if geometry != 1 else None
        frame = frame or host_frames[geometry]
        values = frame.feature("logits").to_numpy().reshape(frame.rows,-1)
        return int(np.argmax(values[0]))

    tokens = tokenizer.chat(prompt) if chat else tokenizer.encode(prompt)
    if not tokens or len(tokens) + max_tokens > model.context:
        raise ValueError("generation exceeds context capacity")
    slot = model.admit()
    generated = []
    try:
        predicted = model.prefill([slot],[tokens],chunk=chunk)
        token = sample(predicted,getattr(model,"last_prefill_chunk",None) or chunk)
        for step in range(max_tokens):
            generated.append(int(token))
            if token == tokenizer.eos or step + 1 == max_tokens:
                break
            token = sample(model.decode([slot], [int(token)]),1)
    finally:
        model.release(slot)
    return dict(
        prompt_tokens=tokens,
        generated_tokens=generated,
        text=tokenizer.decode(generated),
    )
