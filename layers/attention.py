"""GQA Attention layer for GLM-4-9B Dense model.

GLM-4-9B key properties:
  - 32 query heads, 2 KV heads (GQA with ratio 16:1)
  - head_dim = 128
  - attention_bias = True (Q/K/V/O projections have bias)
  - No QK-Norm
  - Standard RoPE (neox style, theta=10000)
"""

import jax
import jax.numpy as jnp
from flax import nnx

from layers.kv_cache import KVCache
from layers.linear import LinearBase
from layers.rotary import RotaryEmbedding


class GQAAttention(nnx.Module):
    """Grouped Query Attention with RoPE.

    Supports both standard MHA (num_kv_heads == num_heads) and
    GQA (num_kv_heads < num_heads) with KV head repetition.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mesh: jax.sharding.Mesh,
        attention_bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim ** -0.5

        # Q/K/V/O projections
        self.q_proj = LinearBase(
            hidden_size, num_heads * head_dim, mesh,
            use_bias=attention_bias, params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.k_proj = LinearBase(
            hidden_size, num_kv_heads * head_dim, mesh,
            use_bias=attention_bias, params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.v_proj = LinearBase(
            hidden_size, num_kv_heads * head_dim, mesh,
            use_bias=attention_bias, params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.o_proj = LinearBase(
            num_heads * head_dim, hidden_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None),
        )

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: KVCache | None = None,
        cache_position: jax.Array | None = None,
    ) -> tuple[jax.Array, KVCache | None]:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            positions: [batch, seq_len] position indices
            attention_mask: [batch, 1, seq_len, kv_len] mask (0=attend, -1e9=mask)
                Prefill: [B, 1, S, max_seq_len] or [B, 1, S, S]
                Decode:  [B, 1, 1, max_seq_len]
            kv_cache: Optional per-layer KV cache. None = no caching.
            cache_position: [seq_len] indices into cache to write new K/V.
                Prefill: [0, 1, ..., prompt_len-1]
                Decode:  [cur_pos]

        Returns:
            (output, kv_cache): output [batch, seq_len, hidden_size],
                updated kv_cache (None if input kv_cache was None).
        """
        batch, seq_len, _ = hidden_states.shape

        # Project Q/K/V
        q = self.q_proj(hidden_states)  # [B, S, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]

        # Reshape to [B, S, num_heads, head_dim]
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE — expects [num_tokens, num_heads, head_dim]
        # Flatten batch for RoPE then reshape back
        q_flat = q.reshape(-1, self.num_heads, self.head_dim)
        k_flat = k.reshape(-1, self.num_kv_heads, self.head_dim)
        pos_flat = positions.reshape(-1)

        q_flat, k_flat = self.rotary_emb(pos_flat, q_flat, k_flat)

        q = q_flat.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k_flat.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # --- KV Cache update ---
        if kv_cache is not None:
            # Write new k, v into cache at cache_position
            k_cache = jax.lax.dynamic_update_slice(
                kv_cache.k, k, (0, cache_position[0], 0, 0),
            )
            v_cache = jax.lax.dynamic_update_slice(
                kv_cache.v, v, (0, cache_position[0], 0, 0),
            )
            kv_cache = KVCache(k=k_cache, v=v_cache)

            # Use full cached K/V for attention
            k_full = k_cache  # [B, max_seq_len, num_kv_heads, head_dim]
            v_full = v_cache
        else:
            k_full = k
            v_full = v

        # Scaled dot-product attention via JAX native implementation.
        # Handles GQA natively (num_kv_heads != num_heads), computes in fp32
        # internally, and supports causal masking.
        # q: [B, S, num_heads, hd], k/v: [B, kv_len, num_kv_heads, hd]

        # Convert additive mask to bias format for dot_product_attention
        attn_output = jax.nn.dot_product_attention(
            q, k_full, v_full,
            bias=attention_mask,
            scale=self.scaling,
            implementation="xla",
        )

        # Reshape: [B, S, num_heads, hd] -> [B, S, num_heads * hd]
        attn_output = attn_output.reshape(batch, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)
        return output, kv_cache
