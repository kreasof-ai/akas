"""Reusable batched runtime for composed floating decoder models.

Weights are shared by slots; all recurrent/cache state is slot-indexed. Prefill
uses an optional chunked graph. Decode replays device sampling/position updates
without host token round trips. No compiler or reference framework is imported.
"""

import json
from pathlib import Path
import numpy as np
from akas.execution.forward import Forward
from akas.memory.transfer import write


class Model:
    def __init__(self, checkpoint, bundle, device, *, graphs=False, tokenizer=True):
        self.device, self.closed = device, False
        self.owned, self.forward = [], None
        self.prefill_forward = None
        manifest = json.loads((Path(bundle) / "forward.json").read_text())
        if manifest.get("provider") != device.info["provider"]:
            raise ValueError("reference bundle requires a different provider")
        self.checkpoint = self.checkpoint_type(checkpoint, tokenizer=tokenizer)
        self.config, self.tokenizer = self.checkpoint.config, self.checkpoint.tokenizer
        meta = manifest.get("reference", {})
        self.batch, self.capacity = meta["batch"], meta["capacity"]
        self.kv_dtype = meta["kv_dtype"]
        self.max_weight_bytes = meta.get("max_weight_bytes", 64 * 1024 * 1024)
        expected = self.definition(
            self.config,
            batch=self.batch,
            capacity=self.capacity,
            provider=manifest["provider"],
            kv_dtype=self.kv_dtype,
            **getattr(self.checkpoint, "definition_options", {}),
            max_weight_bytes=self.max_weight_bytes,
        )
        expected.phase = manifest["forward"]["phase"]
        if json.loads(json.dumps(expected.metadata())) != manifest["forward"]:
            raise ValueError("reference bundle belongs to a different model definition")
        try:
            supplied = self.checkpoint.upload(
                device,
                max_weight_bytes=self.max_weight_bytes,
                **(
                    {"matrix_dtype": meta["matrix_dtype"]}
                    if "matrix_dtype" in meta
                    else {}
                ),
            )
            self.owned.extend(supplied.values())
            for name, spec in expected.buffers.items():
                if spec.role in ("input", "output"):
                    value = device.from_numpy(np.zeros(spec.shape, dtype=spec.dtype))
                    supplied[name] = value
                    self.owned.append(value)
            self.forward = Forward(bundle, device, supplied, graphs=graphs)
            self.buffers = self.forward.buffers
            self.prefill_chunk = meta.get("prefill_chunk", 1)
            if self.prefill_chunk > 1:
                shared = {
                    name: value
                    for name, value in self.buffers.items()
                    if expected.buffers[name].role in ("weight", "state", "output")
                    or name in ("tokens", "active")
                }
                spec = self.definition(
                    self.config,
                    batch=self.batch,
                    capacity=self.capacity,
                    provider=manifest["provider"],
                    kv_dtype=self.kv_dtype,
                    **getattr(self.checkpoint, "definition_options", {}),
                    max_weight_bytes=self.max_weight_bytes,
                    chunk=self.prefill_chunk,
                )
                prefill_manifest = json.loads(
                    (Path(bundle) / "prefill.json").read_text()
                )
                spec.phase = prefill_manifest["forward"]["phase"]
                if (
                    json.loads(json.dumps(spec.metadata()))
                    != prefill_manifest["forward"]
                ):
                    raise ValueError(
                        "prefill bundle belongs to a different model definition"
                    )
                self.owned.append(
                    device.from_numpy(
                        np.zeros(self.batch * self.prefill_chunk, dtype="int32")
                    )
                )
                shared["prefill_tokens"] = self.owned[-1]
                self.prefill_forward = Forward(
                    bundle, device, shared, filename="prefill.json", graphs=graphs
                )
            self.reset()
        except BaseException:
            self.close()
            raise

    @property
    def allocated_bytes(self):
        values = [*self.buffers.values()]
        if self.prefill_forward:
            values.extend(self.prefill_forward.buffers.values())
        return sum(b.nbytes for b in {id(b): b for b in values}.values())

    def _check(self):
        if self.closed:
            raise RuntimeError("reference model is closed")
        self.device._check()

    def reset(self, slots=None):
        self._check()
        self.device.synchronize()
        if slots is None:
            slots = np.arange(self.batch)
        slots = np.asarray(slots)
        if (
            slots.ndim != 1
            or slots.dtype.kind not in "iu"
            or np.any(slots < 0)
            or np.any(slots >= self.batch)
        ):
            raise ValueError("invalid reset slots")
        # Reset convolution state only; fresh positions make stale KV unreachable.
        for name, spec in self.forward.definition.buffers.items():
            if spec.role == "state" and (
                name == "positions" or name.endswith("conv_state")
            ):
                values = self.buffers[name].to_numpy()
                values[slots] = 0
                write(self.device, self.buffers[name], values)
        if not hasattr(self, "positions"):
            self.positions = np.zeros(self.batch, dtype="int32")
        self.positions[slots] = 0
        for name in ("tokens", "sampled", "active"):
            values = self.buffers[name].to_numpy()
            values[slots] = 0
            write(self.device, self.buffers[name], values)

    def _active(self, active):
        active = (
            np.ones(self.batch, dtype="int32") if active is None else np.asarray(active)
        )
        if (
            active.shape != (self.batch,)
            or active.dtype.kind not in "biu"
            or np.any((active != 0) & (active != 1))
        ):
            raise ValueError("active must contain one boolean/0/1 per slot")
        return active.astype("int32")

    def step(self, tokens, *, active=None, read=True):
        self._check()
        tokens = np.asarray(tokens)
        mask = self._active(active)
        if (
            tokens.shape != (self.batch,)
            or tokens.dtype.kind not in "iu"
            or np.any(tokens < 0)
            or np.any(tokens >= self.config.vocab)
        ):
            raise ValueError("invalid batch token IDs")
        if np.any(self.positions[mask != 0] >= self.capacity):
            raise ValueError("reference context exhausted")
        write(self.device, self.buffers["tokens"], tokens.astype("int32"))
        write(self.device, self.buffers["active"], mask)
        self.forward.run()
        self.positions += mask
        return self.logits() if read else None

    def prefill(self, prompts, *, read=True):
        self._check()
        if len(prompts) != self.batch:
            raise ValueError("one prompt required per batch slot")
        encoded = []
        for prompt in prompts:
            if isinstance(prompt, str):
                if self.tokenizer is None:
                    raise ValueError("text prompts require a tokenizer")
                prompt = self.tokenizer.encode(prompt)
            ids = np.asarray(prompt)
            if (
                ids.ndim != 1
                or not len(ids)
                or ids.dtype.kind not in "iu"
                or np.any(ids < 0)
                or np.any(ids >= self.config.vocab)
            ):
                raise ValueError("prompts must contain valid token IDs")
            encoded.append(ids)
        lengths = np.array([len(x) for x in encoded], dtype="int32")
        if np.any(self.positions + lengths > self.capacity):
            raise ValueError("prompt exceeds reference context")
        if self.prefill_forward is not None:
            for start in range(0, int(lengths.max()), self.prefill_chunk):
                counts = np.clip(lengths - start, 0, self.prefill_chunk).astype("int32")
                tokens = np.zeros((self.batch, self.prefill_chunk), dtype="int32")
                for slot, count in enumerate(counts):
                    tokens[slot, :count] = encoded[slot][start : start + count]
                write(
                    self.device,
                    self.prefill_forward.buffers["prefill_tokens"],
                    tokens.reshape(-1),
                )
                write(self.device, self.buffers["active"], counts)
                self.prefill_forward.run()
                self.positions += counts
            return self.logits() if read else None
        # Preserve each slot's last logits/sample when shorter prompts finish.
        last = (
            np.empty((self.batch, self.config.vocab), dtype="float32") if read else None
        )
        for t in range(int(lengths.max())):
            active = t < lengths
            tokens = np.array(
                [ids[t] if t < len(ids) else 0 for ids in encoded], dtype="int32"
            )
            logits = self.step(tokens, active=active, read=read)
            if read:
                last[active] = logits[active]
        return last

    def decode(self, steps, *, active=None, read=True):
        self._check()
        if type(steps) is not int or steps < 1:
            raise ValueError("decode steps must be positive")
        mask = self._active(active)
        if np.any(self.positions[mask != 0] + steps > self.capacity):
            raise ValueError("decode exceeds reference context")
        write(self.device, self.buffers["active"], mask)
        for _ in range(steps):
            self.forward.run()
        self.positions += mask * steps
        return self.buffers["sampled"].to_numpy() if read else None

    def generate(self, prompts, *, max_tokens=128, chat=True):
        from akas.generation.greedy import generate_batch

        single = isinstance(prompts, str)
        results = generate_batch(
            self, [prompts] if single else prompts, max_tokens=max_tokens, chat=chat
        )
        return results[0] if single else results

    def logits(self):
        self._check()
        return self.buffers["logits"].to_numpy().reshape(self.batch, self.config.vocab)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.device.synchronize()
        if self.prefill_forward is not None:
            self.prefill_forward.close()
        if self.forward is not None:
            self.forward.close()
        for b in reversed(self.owned):
            b.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
