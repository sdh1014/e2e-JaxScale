"""Inference runner for GLM-4 models.

Provides simple greedy and sampling generation loops.
Initial version: no KV Cache, full forward pass each step.
"""

import jax
import jax.numpy as jnp


def make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    """Create a causal attention mask.

    Returns:
        mask: [1, 1, seq_len, seq_len] with 0 for attend, -1e9 for masked.
    """
    mask = jnp.triu(jnp.full((seq_len, seq_len), -1e9, dtype=dtype), k=1)
    return mask[None, None, :, :]  # [1, 1, S, S]


def make_positions(seq_len: int, batch_size: int = 1) -> jax.Array:
    """Create position indices [batch, seq_len]."""
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    return jnp.broadcast_to(positions[None, :], (batch_size, seq_len))


def greedy_generate(
    model,
    input_ids: jax.Array,
    max_new_tokens: int = 128,
    eos_token_id: int | None = None,
) -> jax.Array:
    """Simple greedy generation loop (no KV Cache).

    Each step does a full forward pass over the entire sequence.
    This is slow but correct — good for initial validation.

    Args:
        model: GLM4ForCausalLM instance.
        input_ids: [batch, prompt_len] initial token IDs.
        max_new_tokens: Max number of tokens to generate.
        eos_token_id: Stop generation when this token is produced.

    Returns:
        generated_ids: [batch, prompt_len + generated_len]
    """
    current_ids = input_ids

    for _ in range(max_new_tokens):
        seq_len = current_ids.shape[1]
        positions = make_positions(seq_len, current_ids.shape[0])
        mask = make_causal_mask(seq_len, dtype=jnp.bfloat16)

        # Full forward pass
        logits = model(current_ids, positions, mask)

        # Greedy: take argmax of last position
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # [B, 1]

        # Append to sequence
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)

        # Check EOS
        if eos_token_id is not None:
            if jnp.all(next_token == eos_token_id):
                break

    return current_ids


def sample_generate(
    model,
    input_ids: jax.Array,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    eos_token_id: int | None = None,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """Sampling generation loop with temperature and top-k.

    Args:
        model: GLM4ForCausalLM instance.
        input_ids: [batch, prompt_len]
        max_new_tokens: Max new tokens.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        eos_token_id: Stop token.
        rng_key: JAX random key for sampling.

    Returns:
        generated_ids: [batch, prompt_len + generated_len]
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    current_ids = input_ids

    for step in range(max_new_tokens):
        seq_len = current_ids.shape[1]
        positions = make_positions(seq_len, current_ids.shape[0])
        mask = make_causal_mask(seq_len, dtype=jnp.bfloat16)

        logits = model(current_ids, positions, mask)
        next_logits = logits[:, -1, :]  # [B, vocab_size]

        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_logits, top_k)
            # Create mask: -inf for everything outside top-k
            next_logits = jnp.full_like(next_logits, -1e9)
            next_logits = next_logits.at[
                jnp.arange(next_logits.shape[0])[:, None], top_k_indices
            ].set(top_k_logits)

        # Sample
        rng_key, sample_key = jax.random.split(rng_key)
        next_token = jax.random.categorical(sample_key, next_logits, axis=-1)
        next_token = next_token[:, None]  # [B, 1]

        current_ids = jnp.concatenate([current_ids, next_token], axis=1)

        if eos_token_id is not None:
            if jnp.all(next_token == eos_token_id):
                break

    return current_ids
