"""Small independent host oracle for the Q8_0 ZONOS3 backbone."""

import numpy as np

from akas.checkpoints.gguf import dequantize


def _rms(x, epsilon, weight=1):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + epsilon) * weight


def _silu(x):
    return x / (1 + np.exp(-x))


def _softmax(x):
    p = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def reference(checkpoint, embeddings):
    """Evaluate causal GDN, attention residuals and latent MoE in float64."""
    c, weights = checkpoint.config, checkpoint.backbone
    count = len(embeddings)
    positions = np.arange(count)

    def floating(name):
        return weights.array(name, dtype=np.float64)

    def affine(x, name):
        return x @ floating(name).T

    def expert(name, index):
        info = weights.tensors[name]
        size = info.nbytes // info.shape[0]
        raw = weights.packed(name)[index * size : (index + 1) * size]
        return dequantize(raw, info.type).reshape(info.shape[1:]).astype("float64")

    def delta(x, prefix):
        def get(name):
            return floating(prefix + name)

        qkv = affine(x, prefix + "attn_qkv.weight")
        qkv = np.pad(qkv, ((c.conv_width - 1, 0), (0, 0)))
        convolution = get("ssm_conv1d.weight")
        qkv = _silu(
            sum(qkv[i : i + count] * convolution[:, i] for i in range(c.conv_width))
        )
        q, k, v = np.split(
            qkv.reshape(count, 3 * c.delta_heads, c.delta_dim), 3, axis=1
        )
        q = (
            q
            / np.sqrt(np.sum(q**2, axis=-1, keepdims=True) + 1e-6)
            / np.sqrt(c.delta_dim)
        )
        k = k / np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + 1e-6)
        alpha = affine(x, prefix + "ssm_alpha.weight") + get("ssm_dt.bias")
        decay = np.exp(np.logaddexp(0, alpha) * get("ssm_a"))
        beta = 1 / (1 + np.exp(-affine(x, prefix + "ssm_beta.weight")))
        memory = np.zeros((c.delta_heads, c.delta_dim, c.delta_dim), "float64")
        out = []
        for i in range(count):
            memory *= decay[i, :, None, None]
            residual = v[i] - np.einsum("hij,hj->hi", memory, k[i])
            memory += (beta[i, :, None] * residual)[:, :, None] * k[i, :, None, :]
            out.append(np.einsum("hij,hj->hi", memory, q[i]))
        gate = _silu(affine(x, prefix + "attn_gate.weight")).reshape(q.shape)
        normal = _rms(np.asarray(out), c.epsilon, get("ssm_norm.weight"))
        return affine((normal * gate).reshape(count, -1), prefix + "ssm_out.weight")

    def attention(x, prefix):
        d = c.head_dim
        phase = positions[:, None] * c.rope_theta ** (-np.arange(0, d, 2) / d)
        rot = np.exp(1j * phase[:, None, :])

        def rotate(rows):
            uv = (rows[..., 0::2] + 1j * rows[..., 1::2]) * rot
            return np.stack((uv.real, uv.imag), axis=-1).reshape(rows.shape)

        def project(tag, heads):
            return affine(x, prefix + f"attn_{tag}.weight").reshape(count, heads, d)

        q = rotate(
            _rms(project("q", c.heads), 1e-6, floating(prefix + "attn_q_norm.weight"))
        )
        k = rotate(_rms(project("k", c.kv_heads), 1e-6))
        v = project("v", c.kv_heads)
        k, v = (rows.astype("float16").astype("float64") for rows in (k, v))
        k, v = (np.repeat(rows, c.heads // c.kv_heads, axis=1) for rows in (k, v))
        logits = np.einsum("ihd,jhd->hij", q, k) / np.sqrt(d)
        logits = np.where(positions[:, None] >= positions, logits, -np.inf)
        out = np.einsum("hij,jhd->ihd", _softmax(logits), v)
        gate = 1 / (1 + np.exp(-affine(x, prefix + "attn_gate.weight")))
        return affine(
            (out * gate[..., None]).reshape(count, -1), prefix + "attn_output.weight"
        )

    def feed_forward(x, prefix, layer):
        if layer < c.dense_layers:
            gate = affine(x, prefix + "ffn_gate.weight")
            up = affine(x, prefix + "ffn_up.weight")
            return affine(_silu(gate) * up, prefix + "ffn_down.weight")
        raw = affine(_rms(x, c.epsilon), prefix + "ffn_gate_inp.weight")
        selected = np.argsort(-(raw + floating(prefix + "exp_probs_b.bias")), axis=-1)
        selected = selected[:, : c.experts_per_row]
        probabilities = _softmax(np.take_along_axis(raw, selected, axis=-1))
        latent = affine(x, prefix + "ffn_latent_down.weight")
        out = np.zeros_like(latent)
        for row, indexes in enumerate(selected):
            for index, weight in zip(indexes, probabilities[row], strict=True):
                gate, up, down = (
                    expert(prefix + f"ffn_{tag}_exps.weight", index)
                    for tag in ("gate", "up", "down")
                )
                result = down @ (_silu(gate @ latent[row]) * (up @ latent[row]))
                out[row] += weight * result
        return affine(out, prefix + "ffn_latent_up.weight")

    bank = [_rms(embeddings.astype("float64"), c.epsilon)]

    def route(name):
        sources = np.stack(bank, axis=1)
        scores = _rms(sources, c.epsilon) @ floating(name)
        return np.einsum("ts,tsd->td", _softmax(scores), sources)

    for layer, recurrent in enumerate(c.recurrent):
        prefix = f"blk.{layer}."
        x = _rms(
            route(prefix + "attn_res_score.weight"),
            c.epsilon,
            floating(prefix + "attn_norm.weight"),
        )
        bank.append(delta(x, prefix) if recurrent else attention(x, prefix))
        x = _rms(
            route(prefix + "ffn_res_score.weight"),
            c.epsilon,
            floating(prefix + "ffn_norm.weight"),
        )
        bank.append(feed_forward(x, prefix, layer))
    return _rms(
        route("output_res_score.weight"), c.epsilon, floating("output_norm.weight")
    )
