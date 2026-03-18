"""KV Cache for autoregressive inference.

Uses pre-allocated static arrays with dynamic_update_slice for XLA-friendly
in-place updates. Cache is passed as function arguments (not module state)
to follow JAX's functional paradigm.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class KVCache:
    """Per-layer KV Cache.

    Attributes:
        k: [batch, max_seq_len, num_kv_heads, head_dim]
        v: [batch, max_seq_len, num_kv_heads, head_dim]
    """
    k: jax.Array
    v: jax.Array


# Register KVCache as a JAX pytree so it can pass through jax.jit.
jax.tree_util.register_pytree_node(
    KVCache,
    lambda c: ((c.k, c.v), None),
    lambda _, children: KVCache(k=children[0], v=children[1]),
)


def init_kv_caches(
    batch_size: int,
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> list[KVCache]:
    """Pre-allocate KV caches for all layers, initialized to zeros."""
    caches = []
    for _ in range(num_layers):
        k = jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        v = jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        caches.append(KVCache(k=k, v=v))
    return caches
