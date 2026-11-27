"""Independent FP64 NumPy oracle for the Inkling 2 Mini text model.

This is a correctness tool, not a runtime. It re-implements the fixed-pass
Inkling 2 Mini forward (``akas.models.inkling2.model.compose``) in float64,
matching the exact numerical contract of each operation family's independent
``validation.py`` reference — including every intermediate BF16 rounding point,
the ``config.visits()`` loop / residual-bank / cross-pass blend structure, the
relative-attention sink semantics, the dynamic short-conv recurrence, and the
block-scaled (NVFP4 / MXFP8) expert decode.

The oracle qualifies a compiled AKAS program for three-way greedy agreement and
logit RMS noise floor against external baselines. It loads the released
safetensors checkpoint directly and never calls into Tensor codecs or kernels,
so it remains a trustworthy independent reference.

Usage:
    ref = Inkling2Reference("build/checkpoints/inkling-2-mini", context=2048)
    logits = ref.prefill(prompt_token_ids)   # resets state, returns (vocab,)
    logits = ref.decode(last_token_id)       # one step, returns (vocab,)
"""

import numpy as np

from akas.models.inkling2.weights import Inkling2MiniCheckpoint, _PREFIX


def _bf16(x):
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    return decode_bfloat16(encode_bfloat16(np.asarray(x, dtype="float32")))


class _LazyWeights:
    """Mapping facade that decodes each checkpoint tensor on first access."""

    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, short):
        return self.owner._weight(short)


class Inkling2Reference:
    """One-request FP64 forward over the released Inkling 2 Mini checkpoint."""

    def __init__(self, path, *, context=2048):
        self.checkpoint = Inkling2MiniCheckpoint(path, tokenizer=False)
        self.config = self.checkpoint.config
        self.context = context
        self._cache = {}
        self.reset()

    # ---- checkpoint loading (lazy, per-weight) -------------------------------

    @property
    def weights(self):
        return _LazyWeights(self)

    def _read(self, short):
        from akas.checkpoints.safetensors import read_tensor
        from tensor.runtime.dtypes import decode_bfloat16

        name = _PREFIX + short
        info = self.checkpoint._tensors[name]
        raw = read_tensor(info)
        if info.dtype == "BF16":
            return decode_bfloat16(raw).astype("float64")
        if info.dtype in ("F8_E4M3", "U8"):
            # Keep raw storage bits; expert/quant decode interprets them.
            return np.asarray(raw, dtype="uint8")
        if info.dtype == "F32":
            return np.asarray(raw, dtype="float64")
        if info.dtype == "F16":
            return np.asarray(raw, dtype="float16").astype("float64")
        raise ValueError(f"unsupported storage dtype {info.dtype}")

    def _weight(self, short):
        if short not in self._cache:
            self._cache[short] = self._read(short)
        return self._cache[short]

    def _w(self, name):
        """A BF16 dense weight, already decoded to float64 at load time."""
        return self.weights[name]

    def _wf(self, name):
        """A native F32 weight (mix queries / router bias), full precision."""
        return self.weights[name]

    # ---- state --------------------------------------------------------------

    def reset(self):
        c = self.config
        self.position = 0
        self.kv = {}
        self.conv = {}
        kv_width = c.kv_heads * c.head_dim
        for layer, pass_index in c.visits():
            local = layer in c.local_layers
            capacity = min(self.context, c.window) if local else self.context
            self.kv[(layer, pass_index)] = (
                np.zeros((capacity, c.kv_heads, c.head_dim), np.float64),
                np.zeros((capacity, c.kv_heads, c.head_dim), np.float64),
            )
            self.conv[("attn", layer, pass_index)] = np.zeros((c.taps - 1, c.width))
            self.conv[("mlp", layer, pass_index)] = np.zeros((c.taps - 1, c.width))
            self.conv[("k", layer, pass_index)] = np.zeros((c.taps - 1, kv_width))
            self.conv[("v", layer, pass_index)] = np.zeros((c.taps - 1, kv_width))

    # ---- primitive ops (validation.py references) ----------------------------

    def _linear(self, x, name):
        return _bf16(x).astype("float64") @ self._w(name).T

    def _rms(self, x, name, out_bf16):
        c = self.config
        xf = np.asarray(x, dtype="float64")
        gain = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + c.epsilon)
        out = xf * gain * self._w(name)
        return _bf16(out).astype("float64") if out_bf16 else out

    def _head_norm(self, x, heads, depth, *, scale=None, clip=None, positions=None):
        c = self.config
        src = _bf16(x.reshape(x.shape[0], heads, depth)).astype("float64")
        if scale is not None:
            src = src * self._w(scale)
        square = np.sum(src**2, axis=-1, keepdims=True)
        if clip is None:
            gain = 1.0 / np.sqrt(square / depth + c.epsilon)
        else:
            gain = np.minimum(clip / (np.sqrt(square) + c.epsilon), 1.0)
        out = _bf16(src * gain).astype("float64")
        if positions is not None:
            tau = 1.0 + c.log_alpha * np.log(
                np.maximum((positions + 1.0) / c.log_floor, 1.0)
            )
            out = _bf16(out * tau[:, None, None]).astype("float64")
        return out.reshape(x.shape)

    def _short_conv(self, x, history, *, weight, dynamic_w=None):
        c = self.config
        rounded = _bf16(x).astype("float64")
        channels = rounded.shape[-1]
        coef = weight.T[::-1].T if dynamic_w is None else weight.T  # (channels, taps)
        out = np.empty_like(rounded)
        hist = history.copy()
        for row in range(rounded.shape[0]):
            current = rounded[row]
            window = np.concatenate((current[None], hist), axis=0)
            kernel = coef
            if dynamic_w is not None:
                dw = _bf16(dynamic_w[row]).astype("float64")
                groups = dw.shape[0] // c.taps
                kernel = coef + np.repeat(per := dw.reshape(groups, c.taps),
                                          channels // groups, axis=0)
            acc = current + np.einsum("tc,ct->c", window, kernel)
            divisor = np.sqrt(1.0 + c.taps * (2 if dynamic_w is not None else 1))
            out[row] = _bf16(acc / divisor).astype("float64")
            hist = window[:-1]
        return out, hist

    # ---- attention ------------------------------------------------------------

    def _attention(self, streams, source, key, local, positions):
        c = self.config
        qx, kx, vx = streams
        h, kh, d = c.heads, c.kv_heads, c.head_dim
        capacity = min(self.context, c.window) if local else self.context
        extent = c.window if local else c.relative_extent
        count = qx.shape[0]
        kcache, vcache = self.kv[key]

        q = self._head_norm(
            self._linear(qx, source + "wq_du.weight"), h, d,
            scale=source + "q_l2_clip.scale", clip=c.qk_clip,
            positions=None if local else positions,
        )
        k = self._linear(kx, source + "wk_dv.weight")
        v = self._linear(vx, source + "wv_dv.weight")
        for stream in ("k", "v"):
            proj = k if stream == "k" else v
            driver = kx if stream == "k" else vx
            conv_src = source + f"{stream}_sconv."
            if local:
                weight = self._w(conv_src + "weight").reshape(kh * d, c.taps)
                dyn = None
            else:
                weight = self._w(conv_src + "static_w")  # (taps, kv)
                dyn = self._linear(driver, conv_src + "weight_proj")
            proj, self.conv[(stream, *key)] = self._short_conv(
                proj, self.conv[(stream, *key)], weight=weight, dynamic_w=dyn
            )
            if stream == "k":
                k = proj
            else:
                v = proj
        k = self._head_norm(k, kh, d, clip=c.qk_clip).reshape(count, kh, d)
        v = self._head_norm(v, kh, d).reshape(count, kh, d)
        rel = self._head_norm(
            self._linear(qx, source + "wr_du.weight"), h, c.relative_dim,
            scale=source + "rel_l2_clip.scale", clip=c.relative_clip,
            positions=None if local else positions,
        ).reshape(count, h, c.relative_dim)
        sinks = self._linear(qx, source + "attn_sink_proj.weight")
        bias = self._w(source + "rel_logits_proj.proj")  # (relative_dim, extent)

        q = q.reshape(count, h, d)
        attended = np.zeros((count, h, d), np.float64)
        for row in range(count):
            position = int(positions[row])
            live = np.arange(max(0, position + 1 - capacity), position + 1)
            # Earlier rows of this same chunk are already committed (below), so
            # their k/v come from the cache; only the current row is read fresh.
            past = live[live < position]
            for head in range(h):
                kv_head = head // (h // kh)
                keys = np.concatenate(
                    (kcache[past % capacity, kv_head], k[[row], kv_head])
                )
                vecs = np.concatenate(
                    (vcache[past % capacity, kv_head], v[[row], kv_head])
                )
                logits = keys @ q[row, head] / np.sqrt(d)
                distances = position - live
                mask = distances < extent
                if mask.any():
                    logits[mask] += rel[row, head] @ bias[:, distances[mask]]
                peak = max(logits.max(), sinks[row, head])
                prob = np.exp(logits - peak)
                attended[row, head] = prob @ vecs / (
                    prob.sum() + np.exp(sinks[row, head] - peak)
                )
        # Commit this chunk's k/v rows at their absolute positions.
        for row in range(count):
            kcache[int(positions[row]) % capacity] = k[row]
            vcache[int(positions[row]) % capacity] = v[row]
        gate = self._linear(qx, source + "gate_attn.weight")
        gated = attended.reshape(count, h * d) * (1.0 / (1.0 + np.exp(-gate)))
        return self._linear(gated, source + "wo_ud.weight")

    # ---- MoE -------------------------------------------------------------------

    def _moe(self, x, source):
        c = self.config
        count = x.shape[0]
        experts, top, middle = c.experts, c.top_k, c.expert_width
        logits = self._linear(x, source + "gate.weight")
        scores = 1.0 / (1.0 + np.exp(-logits))
        order = np.argsort(-(scores + self._wf(source + "gate.bias")), axis=1,
                           kind="stable")[:, :top]
        chosen = np.take_along_axis(scores, order, axis=1)
        weights = chosen / (chosen.sum(axis=1, keepdims=True) + 1e-6) * c.route_scale

        def project(input_, part, columns, depth, routed):
            name = source + f"experts.{part}_weight"
            kind = self.checkpoint.expert_format(name)
            if kind == "nvfp4":
                from akas.quantization.weights.nvfp4.validation import decode_rows
            else:
                from akas.quantization.weights.mxfp8.validation import decode_rows

            # Decode each unique expert used by this batch exactly once.
            unique = np.unique(order)
            decoded = {}
            denom = (
                float(self.weights[name + "_outer_scale_denominator"][0])
                if kind == "nvfp4"
                else None
            )
            for e in unique:
                e = int(e)
                if kind == "nvfp4":
                    w = decode_rows(
                        self.weights[name][e],
                        self.weights[name + "_scale"][e],
                        float(self.weights[name + "_amax"][e]),
                        denom,
                    )
                else:
                    w = decode_rows(
                        self.weights[name][e], self.weights[name + "_scale"][e]
                    )
                decoded[e] = _bf16(w).astype("float64")
            out = np.zeros((count, top, columns), np.float64)
            for row in range(count):
                for rank in range(top):
                    e = int(order[row, rank])
                    src = input_[row, rank] if routed else input_[row]
                    out[row, rank] = src @ decoded[e].T
            return out

        gate_up = project(x, "w13", 2 * middle, c.width, False)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        cg = c.gate_cap * np.tanh(gate / c.gate_cap)
        cu = c.up_cap * np.tanh(up / c.up_cap)
        act = _bf16(cg / (1.0 + np.exp(-gate)) * cu * weights[..., None]).astype(
            "float64"
        )
        down = project(act, "w2", c.width, middle, True)
        return _bf16(down.sum(axis=1)).astype("float64")

    # ---- residual bank / mix / blend ------------------------------------------

    @staticmethod
    def _bank(sources):
        return np.stack(list(sources), axis=1)

    def _mix(self, stack, query, index):
        c = self.config
        normalized = stack / np.sqrt(
            np.mean(stack**2, axis=-1, keepdims=True) + c.epsilon
        )
        q = self._wf(query).reshape(-1, c.width)  # (n_queries, width)
        scores = normalized @ q[index] / np.sqrt(c.width)
        gates = np.exp(scores - scores.max(axis=-1, keepdims=True))
        gates /= gates.sum(axis=-1, keepdims=True)
        return np.einsum("rb,rbc->rc", gates, normalized)

    def _blend(self, x, previous, layer, stream, source):
        c = self.config
        if stream in ("k", "v"):
            query = "middle_cycle_cross_pass_kv_queries"
            index = 2 * (layer - c.loop_start) + (stream == "v")
        else:
            query = "middle_cycle_cross_pass_queries"
            index = 2 * (layer - c.loop_start) + (stream == "mlp")
        message = self._mix(previous, query, index)
        left = self._head_norm(x, 1, c.width)
        right = self._head_norm(message, 1, c.width)
        control = source + f"controllers.{stream}."
        rank = self._linear(np.concatenate((left, right), axis=1),
                            control + "down_weight")
        forget = self._linear(rank, control + "forget_weight")
        gain = self._linear(rank, control + "gain_weight")
        return (1 - np.tanh(forget)) * x + np.tanh(gain) * message

    # ---- forward -----------------------------------------------------------------

    def _forward(self, tokens, positions):
        c = self.config
        width = c.width
        tokens = np.asarray(tokens, dtype="int64")
        embedded = self.weights["embed.weight"][tokens]
        partial = self._rms(embedded, "embed_norm.weight", out_bf16=False)
        blocks = ()
        prelude = previous = None

        for layer, pass_index in c.visits():
            source = f"layers.{layer}."
            key = (layer, pass_index)
            if layer == c.loop_start:
                if pass_index == 0:
                    prelude = (blocks, partial)
                else:
                    previous = self._bank((*blocks[len(prelude[0]) + 1:], partial))
                    blocks, partial = prelude
            if layer == c.loop_end:
                previous = None

            local = layer in c.local_layers
            if layer:
                stack = self._bank((*blocks, partial))
                qmix = self._mix(stack, f"attn_res.attn_queries.{layer - 1}", 0)
                streams = (
                    (qmix, qmix, qmix)
                    if local
                    else (
                        qmix,
                        self._mix(stack, f"attn_res.attn_queries.{layer - 1}", 1),
                        self._mix(stack, f"attn_res.attn_queries.{layer - 1}", 2),
                    )
                )
            else:
                streams = (partial, partial, partial)
            if previous is not None:
                streams = tuple(
                    self._blend(x, previous, layer, sname, source)
                    for x, sname in zip(streams, ("attn", "k", "v"))
                )
            if local and previous is None:
                inp = self._rms(streams[0], source + "attn_norm.weight", out_bf16=True)
                inputs = (inp, inp, inp)
            else:
                inputs = tuple(
                    self._rms(
                        x,
                        source + ("attn_norm.weight" if local
                                  else f"{sname}_input_norm.weight"),
                        out_bf16=True,
                    )
                    for x, sname in zip(streams, ("q", "k", "v"))
                )
            attended = self._attention(inputs, source + "attn.", key, local, positions)
            attended, self.conv[("attn", *key)] = self._short_conv(
                attended, self.conv[("attn", *key)],
                weight=self._w(source + "attn_sconv.weight").reshape(width, c.taps),
            )
            if layer % c.block_size == 0:
                blocks = (*blocks, partial)
                residual = attended
            else:
                residual = partial + attended

            mlp_bank = self._bank((*blocks, residual))
            inp = self._mix(mlp_bank, f"attn_res.mlp_queries.{layer}", 0)
            if previous is not None:
                inp = self._blend(inp, previous, layer, "mlp", source)
            inp = self._rms(inp, source + "mlp_norm.weight", out_bf16=True)
            mixed = self._moe(inp, source + "mlp.")
            mixed, self.conv[("mlp", *key)] = self._short_conv(
                mixed, self.conv[("mlp", *key)],
                weight=self._w(source + "mlp_sconv.weight").reshape(width, c.taps),
            )
            partial = residual + mixed

        final = self._mix(self._bank((*blocks, partial)), "attn_res.output_query", 0)
        hidden = self._rms(final, "norm.weight", out_bf16=True)
        raw = self._linear(hidden, "unembed.weight")
        return c.logit_cap * np.tanh(raw / c.logit_cap)

    def prefill(self, tokens):
        """Reset state, run a prompt, and return the last token's logits."""
        self.reset()
        tokens = np.asarray(tokens, dtype="int64")
        positions = np.arange(len(tokens), dtype="int32")
        logits = self._forward(tokens, positions)
        self.position = len(tokens)
        return logits[-1, : self.config.vocab]

    def decode(self, token):
        """Run one decode step for the current request and return its logits."""
        position = np.array([self.position], dtype="int32")
        logits = self._forward(np.array([token], dtype="int64"), position)
        self.position += 1
        return logits[-1, : self.config.vocab]
