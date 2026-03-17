"""Rotary Position Embedding (RoPE) for GLM-4 models."""

import jax
import jax.numpy as jnp
import numpy as np


class RotaryEmbedding:
    """RoPE implementation for GLM-4.

    GLM-4-9B uses partial rotation: only the first half of each head
    (rotary_dim = head_dim // 2 = 64) is rotated, the rest passes through.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        dtype: jnp.dtype = jnp.bfloat16,
        partial_rotary_factor: float = 0.5,
    ):
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.dtype = dtype

        # Precompute inverse frequencies (numpy, no JAX trace)
        # HF uses arange(0, rotary_dim, 2) / rotary_dim
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim)
        )
        self._inv_freq = inv_freq  # shape: (rotary_dim // 2,)

    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply RoPE to query and key tensors.

        Args:
            positions: [num_tokens] position indices.
            query: [num_tokens, num_heads, head_dim]
            key: [num_tokens, num_kv_heads, head_dim]

        Returns:
            Rotated (query, key) with same shapes.
        """
        positions = positions.flatten()
        inv_freq = jnp.asarray(self._inv_freq, dtype=jnp.float32)

        # freqs: [num_tokens, rotary_dim // 2]
        freqs = jnp.einsum("n,d->nd", positions.astype(jnp.float32), inv_freq)
        cos = jnp.cos(freqs).astype(self.dtype)
        sin = jnp.sin(freqs).astype(self.dtype)

        query = _apply_rotary_emb(query, cos, sin, self.rotary_dim)
        key = _apply_rotary_emb(key, cos, sin, self.rotary_dim)
        return query, key


def _apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    rotary_dim: int,
) -> jax.Array:
    """Apply neox-style rotary embedding with partial rotation.

    Args:
        x: [num_tokens, num_heads, head_dim]
        cos: [num_tokens, rotary_dim // 2]
        sin: [num_tokens, rotary_dim // 2]
        rotary_dim: Number of dimensions to rotate.
    """
    # Split into rotated and pass-through parts
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Expand for broadcasting over heads: [num_tokens, 1, rotary_dim // 2]
    cos = jnp.expand_dims(cos, axis=1).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=1).astype(x.dtype)

    # Neox style: split rotated part in half
    x1, x2 = jnp.split(x_rot, 2, axis=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    x_rotated = jnp.concatenate((o1, o2), axis=-1)

    return jnp.concatenate((x_rotated, x_pass), axis=-1)
