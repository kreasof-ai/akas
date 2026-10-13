"""Independent projection and fused FFN error bounds for kernel tests."""

import numpy as np

def oracle(inputs, weight):
    lhs = inputs.astype(np.float16).astype(np.float64)
    rhs = weight.astype(np.float64)
    return lhs @ rhs.T, np.abs(lhs) @ np.abs(rhs).T * 3e-6 + 1e-10

def reference(x, weights):
    g, gb = oracle(x, weights[0])
    if len(weights) == 1:
        return g, gb
    u, ub = oracle(x, weights[1])
    silu = g / (1 + np.exp(-g))
    expected = silu * u
    bound = (
        np.abs(u) * 1.1 * gb
        + np.abs(silu) * ub
        + 1.1 * gb * ub
        + np.abs(expected) * 2e-6
        + 1e-10
    )
    return expected, bound
