"""Multi-head Latent Attention (MLA) for GLM-4.7-Flash.

MLA compresses Q and KV through low-rank projections to reduce KV cache size.
RoPE is applied only to a subset of head dimensions (qk_rope_head_dim).

GLM-4.7-Flash key properties:
  - 20 attention heads, num_kv_heads=20 (MHA, not GQA)
  - q_lora_rank=768, kv_lora_rank=512
  - qk_nope_head_dim=192, qk_rope_head_dim=64 → full head = 256
  - v_head_dim=256
  - attention_bias=False
  - RoPE theta=1e6, partial_rotary_factor=1.0 (full rotation on rope dims)

Weight structure:
  q_a_proj:             hidden → q_lora_rank           (compression)
  q_a_layernorm:        RMSNorm(q_lora_rank)
  q_b_proj:             q_lora_rank → H*(nope+rope)    (decompression, column-parallel)
  kv_a_proj_with_mqa:   hidden → kv_lora_rank + rope   (compression + shared rope key)
  kv_a_layernorm:       RMSNorm(kv_lora_rank)
  kv_b_proj:            kv_lora_rank → H*(nope+v)      (decompression, column-parallel)
  o_proj:               H*v → hidden                   (row-parallel)
"""

import jax
import jax.numpy as jnp
from flax import nnx

from layers.kv_cache import KVCache
from layers.linear import LinearBase
from layers.normalization import RMSNorm
from layers.rotary import RotaryEmbedding


class MLAAttention(nnx.Module):
    """Multi-head Latent Attention with RoPE on a subset of dimensions."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 256
        self.scaling = self.qk_head_dim ** -0.5

        # Q path: hidden → compressed → per-head Q
        self.q_a_proj = LinearBase(
            hidden_size, q_lora_rank, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, None),  # compression, replicated
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype)
        self.q_b_proj = LinearBase(
            q_lora_rank, num_heads * (qk_nope_head_dim + qk_rope_head_dim), mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"),  # column-parallel
        )

        # KV path: hidden → compressed (+rope key) → per-head KV
        self.kv_a_proj_with_mqa = LinearBase(
            hidden_size, kv_lora_rank + qk_rope_head_dim, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, None),  # compression, replicated
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype)
        self.kv_b_proj = LinearBase(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"),  # column-parallel
        )

        # Output projection
        self.o_proj = LinearBase(
            num_heads * v_head_dim, hidden_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None),  # row-parallel
        )

        # RoPE for the rope portion only (full rotation, partial_rotary_factor=1.0)
        self.rotary_emb = RotaryEmbedding(
            head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            dtype=dtype,
            partial_rotary_factor=1.0,
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
            positions: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, kv_len]
            kv_cache: KV cache with k=[B, max_seq, H, qk_head_dim], v=[B, max_seq, H, v_head_dim]
            cache_position: [seq_len] write indices.

        Returns:
            (output, kv_cache)
        """
        batch, seq_len, _ = hidden_states.shape

        # --- Q path ---
        q_compressed = self.q_a_proj(hidden_states)       # [B, S, q_lora_rank]
        q_compressed = self.q_a_layernorm(q_compressed)
        q = self.q_b_proj(q_compressed)                   # [B, S, H*(nope+rope)]
        q = q.reshape(batch, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = jnp.split(q, [self.qk_nope_head_dim], axis=-1)

        # --- KV path ---
        kv_combined = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_lora+rope]
        kv_compressed, k_rope_shared = jnp.split(
            kv_combined, [self.kv_a_layernorm.hidden_size], axis=-1
        )
        kv_compressed = self.kv_a_layernorm(kv_compressed)    # [B, S, kv_lora]
        kv = self.kv_b_proj(kv_compressed)                    # [B, S, H*(nope+v)]
        kv = kv.reshape(batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)

        # Expand shared rope key to all heads: [B, S, 1, rope] → [B, S, H, rope]
        k_rope_shared = k_rope_shared[:, :, None, :]  # [B, S, 1, rope]
        k_rope = jnp.broadcast_to(k_rope_shared, (batch, seq_len, self.num_heads, self.qk_rope_head_dim))

        # --- RoPE (only on rope dimensions) ---
        # Flatten batch for RoPE: expects [num_tokens, num_heads, rope_dim]
        q_rope_flat = q_rope.reshape(-1, self.num_heads, self.qk_rope_head_dim)
        k_rope_flat = k_rope.reshape(-1, self.num_heads, self.qk_rope_head_dim)
        pos_flat = positions.reshape(-1)

        q_rope_flat, k_rope_flat = self.rotary_emb(pos_flat, q_rope_flat, k_rope_flat)

        q_rope = q_rope_flat.reshape(batch, seq_len, self.num_heads, self.qk_rope_head_dim)
        k_rope = k_rope_flat.reshape(batch, seq_len, self.num_heads, self.qk_rope_head_dim)

        # Concatenate nope and rope parts
        q = jnp.concatenate([q_nope, q_rope], axis=-1)  # [B, S, H, nope+rope]
        k = jnp.concatenate([k_nope, k_rope], axis=-1)  # [B, S, H, nope+rope]

        # --- KV Cache ---
        if kv_cache is not None:
            k_cache = jax.lax.dynamic_update_slice(
                kv_cache.k, k, (0, cache_position[0], 0, 0),
            )
            v_cache = jax.lax.dynamic_update_slice(
                kv_cache.v, v, (0, cache_position[0], 0, 0),
            )
            kv_cache = KVCache(k=k_cache, v=v_cache)
            k_full = k_cache
            v_full = v_cache
        else:
            k_full = k
            v_full = v

        # --- Attention ---
        # [B, H, S, dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k_full = jnp.transpose(k_full, (0, 2, 1, 3))
        v_full = jnp.transpose(v_full, (0, 2, 1, 3))

        attn_weights = jnp.matmul(q, jnp.swapaxes(k_full, -2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(v_full.dtype)
        attn_output = jnp.matmul(attn_weights, v_full)  # [B, H, S, v_head_dim]

        # Reshape back: [B, H, S, v] → [B, S, H*v]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch, seq_len, -1)

        output = self.o_proj(attn_output)
        return output, kv_cache
