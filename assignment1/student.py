"""
Negacyclic Number Theoretic Transform (NTT) implementation.

The negacyclic NTT computes

    y[k] = Σ x[n] · ψ^((2k + 1)n) (mod q)

for a primitive 2N-th root ψ. We use the standard twist trick:

1. Twist the input by ψ^n.
2. Run a cyclic radix-2 NTT with ω = ψ².
3. Convert the Cooley-Tukey bit-reversed output back to normal order.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _as_u32(x):
    return jnp.asarray(x, dtype=jnp.uint32)


def _bit_reverse_indices(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        x = i
        rev = 0
        for _ in range(bits):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        out[i] = rev
    return out


# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    q = jnp.asarray(q, dtype=jnp.uint32)
    s = a + b
    return jnp.where(s >= q, s - q, s).astype(jnp.uint32)


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    q = jnp.asarray(q, dtype=jnp.uint32)
    return jnp.where(a >= b, a - b, a + q - b).astype(jnp.uint32)


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    prod = a.astype(jnp.uint64) * b.astype(jnp.uint64)
    return jnp.remainder(prod, q64).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------

def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT for a batch of inputs.

    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % (2N) == 0
        psi_powers: ψ^n table, shape (N,)
        twiddles: Either the raw twiddle table or (twiddle_table, bitrev_indices)

    Returns:
        jnp.ndarray: NTT output, same shape as input, dtype uint32
    """
    if isinstance(twiddles, tuple):
        twiddle_table, bitrev = twiddles
    else:
        twiddle_table = twiddles
        bitrev = _as_u32(_bit_reverse_indices(x.shape[1]))

    a = mod_mul(_as_u32(x), _as_u32(psi_powers), q)
    batch, n = a.shape

    span = 1
    while span < n:
        w = _as_u32(twiddle_table[span : 2 * span]).reshape(1, 1, span)
        a = a.reshape(batch, n // (2 * span), 2 * span)
        left = a[:, :, :span]
        right = a[:, :, span:]
        t = mod_mul(right, w, q)
        a = jnp.concatenate((mod_add(left, t, q), mod_sub(left, t, q)), axis=2)
        span <<= 1

    return a.reshape(batch, n)[:, bitrev].astype(jnp.uint32)


def prepare_tables(*, q, psi_powers, twiddles):
    """
    Optional one-time table preparation.

    The benchmark excludes this cost, so we precompute the final bit-reversal
    permutation once and reuse the provided stage twiddles directly.
    """
    n = int(psi_powers.shape[0])
    bitrev = jnp.asarray(_bit_reverse_indices(n), dtype=jnp.int32)
    return _as_u32(psi_powers), (_as_u32(twiddles), bitrev)