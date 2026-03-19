"""Inference runner for GLM-4 models.

All generation loops use KV Cache by default.
JittedModel wraps a Flax NNX model with jax.jit for compiled inference.
"""

import logging
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from layers.kv_cache import KVCache, init_kv_caches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    """Create a causal attention mask (no cache, for standalone forward pass).

    Returns:
        mask: [1, 1, seq_len, seq_len] with 0 for attend, -1e9 for masked.
    """
    mask = jnp.triu(jnp.full((seq_len, seq_len), -1e9, dtype=dtype), k=1)
    return mask[None, None, :, :]  # [1, 1, S, S]


def make_prefill_mask(
    seq_len: int,
    max_cache_len: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Prefill mask sized for the pre-allocated cache.

    Returns: [1, 1, seq_len, max_cache_len]
    Left seq_len columns are causal, remaining columns are masked out (padding).
    """
    causal = jnp.triu(jnp.full((seq_len, seq_len), -1e9, dtype=dtype), k=1)
    padding = jnp.full((seq_len, max_cache_len - seq_len), -1e9, dtype=dtype)
    mask = jnp.concatenate([causal, padding], axis=1)
    return mask[None, None, :, :]


def make_decode_mask(
    cache_position: int,
    max_cache_len: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Decode mask: current token attends to cache positions [0, cache_position].

    Returns: [1, 1, 1, max_cache_len]
    """
    mask = jnp.where(
        jnp.arange(max_cache_len) <= cache_position,
        jnp.zeros(max_cache_len, dtype=dtype),
        jnp.full(max_cache_len, -1e9, dtype=dtype),
    )
    return mask[None, None, None, :]


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def make_positions(seq_len: int, batch_size: int = 1) -> jax.Array:
    """Create position indices [batch, seq_len]."""
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    return jnp.broadcast_to(positions[None, :], (batch_size, seq_len))


# ---------------------------------------------------------------------------
# Internal: KV Cache init helper
# ---------------------------------------------------------------------------

def _init_caches(model, batch_size: int, max_cache_len: int):
    """Create empty KV caches from model config.

    Handles both GQA (GLM-4-9B) and MLA (GLM-4.7-Flash) cache shapes.
    For MLA: K=[B,S,H,qk_head_dim], V=[B,S,H,v_head_dim] (both 256 for Flash).
    """
    config = model.config
    mesh = getattr(model, 'mesh', None)

    if config.is_mla:
        # MLA: K uses qk_nope+qk_rope, V uses v_head_dim
        # For GLM-4.7-Flash both are 256, so head_dim works for both
        num_kv_heads = config.num_attention_heads
        head_dim = config.head_dim  # qk_nope + qk_rope = 256
    else:
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

    return init_kv_caches(
        batch_size=batch_size,
        max_seq_len=max_cache_len,
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=model.dtype,
        mesh=mesh,
    )


# ---------------------------------------------------------------------------
# JIT-compiled model wrapper
# ---------------------------------------------------------------------------

class JittedModel:
    """Wraps a Flax NNX model with jax.jit for compiled inference.

    Uses the nnx.split/merge pattern from sglang-jax:
    1. Split model into graph definition + parameter state
    2. Flatten state to leaves for JIT compatibility
    3. Inside JIT: unflatten → merge → call → return

    Two JIT cache entries are created automatically:
    - Prefill: input shape [B, prompt_len]
    - Decode:  input shape [B, 1]
    """

    def __init__(self, model):
        self.config = model.config
        self.dtype = model.dtype
        self.mesh = getattr(model, 'mesh', None)

        # NNX split: separate graph structure from parameter state
        model_def, model_state = nnx.split(model)
        self.model_def = model_def
        self.model_state_leaves, self._model_state_def = jax.tree_util.tree_flatten(
            model_state
        )

        # JIT-compiled forward
        @partial(
            jax.jit,
            donate_argnames=["kv_caches"],
            static_argnames=["model_state_def"],
        )
        def _forward(
            model_def,
            model_state_def,
            model_state_leaves,
            input_ids,
            positions,
            attention_mask,
            kv_caches,
            cache_position,
        ):
            model_state = jax.tree_util.tree_unflatten(
                model_state_def, model_state_leaves
            )
            model = nnx.merge(model_def, model_state)
            return model(
                input_ids,
                positions,
                attention_mask,
                kv_caches=kv_caches,
                cache_position=cache_position,
            )

        self._forward = _forward

    def __call__(
        self,
        input_ids,
        positions,
        attention_mask,
        kv_caches=None,
        cache_position=None,
    ):
        # Non-cached path (debug/validation): skip JIT
        if kv_caches is None:
            model_state = jax.tree_util.tree_unflatten(
                self._model_state_def, self.model_state_leaves
            )
            model = nnx.merge(self.model_def, model_state)
            return model(input_ids, positions, attention_mask)

        return self._forward(
            self.model_def,
            self._model_state_def,
            self.model_state_leaves,
            input_ids,
            positions,
            attention_mask,
            kv_caches,
            cache_position,
        )

    def warmup(self, batch_size=1, prompt_len=128, max_cache_len=256):
        """Precompile JIT for prefill and decode shapes.

        Runs dummy forward passes to trigger XLA compilation before real inference.
        """
        logger.info(
            "JIT warmup: batch=%d, prompt=%d, cache=%d",
            batch_size, prompt_len, max_cache_len,
        )

        kv_caches = _init_caches(self, batch_size, max_cache_len)

        # 1. Prefill compilation
        logger.info("  Compiling prefill (seq_len=%d)...", prompt_len)
        dummy_ids = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
        positions = make_positions(prompt_len, batch_size)
        cache_pos = jnp.arange(prompt_len, dtype=jnp.int32)
        mask = make_prefill_mask(prompt_len, max_cache_len, dtype=self.dtype)

        logits, kv_caches = self(
            dummy_ids, positions, mask,
            kv_caches=kv_caches, cache_position=cache_pos,
        )
        jax.block_until_ready(logits)
        logger.info("  Prefill compiled.")

        # 2. Decode compilation
        logger.info("  Compiling decode (seq_len=1)...")
        decode_ids = jnp.ones((batch_size, 1), dtype=jnp.int32)
        positions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        cache_pos = jnp.array([prompt_len], dtype=jnp.int32)
        mask = make_decode_mask(prompt_len, max_cache_len, dtype=self.dtype)

        logits, kv_caches = self(
            decode_ids, positions, mask,
            kv_caches=kv_caches, cache_position=cache_pos,
        )
        jax.block_until_ready(logits)
        logger.info("  Decode compiled. Warmup done.")


# ---------------------------------------------------------------------------
# Greedy generation (default, with KV Cache)
# ---------------------------------------------------------------------------

def greedy_generate(
    model,
    input_ids: jax.Array,
    max_new_tokens: int = 128,
    max_cache_len: int | None = None,
    eos_token_id: int | list[int] | None = None,
) -> jax.Array:
    """Greedy generation with KV Cache.

    Prefill processes the full prompt in one pass, then decode generates
    one token at a time reusing cached K/V.

    Args:
        model: GLM4ForCausalLM instance.
        input_ids: [batch, prompt_len] initial token IDs.
        max_new_tokens: Max number of tokens to generate.
        max_cache_len: Pre-allocated cache length. Defaults to prompt_len + max_new_tokens.
        eos_token_id: Stop generation when this token is produced.
            Can be a single int or a list of ints (e.g. GLM-4 has 3 EOS tokens).

    Returns:
        generated_ids: [batch, prompt_len + generated_len]
    """
    batch_size, prompt_len = input_ids.shape
    if max_cache_len is None:
        max_cache_len = prompt_len + max_new_tokens

    # 1. Init KV Cache
    kv_caches = _init_caches(model, batch_size, max_cache_len)

    # 2. Prefill: process entire prompt
    positions = make_positions(prompt_len, batch_size)
    cache_position = jnp.arange(prompt_len, dtype=jnp.int32)
    mask = make_prefill_mask(prompt_len, max_cache_len, dtype=model.dtype)

    logits, kv_caches = model(
        input_ids, positions, mask,
        kv_caches=kv_caches, cache_position=cache_position,
    )

    # Take last-position logits
    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # [B, 1]
    generated = [next_token]

    # 3. Decode: generate one token at a time
    for step in range(max_new_tokens - 1):
        cur_pos = prompt_len + step

        positions = jnp.full((batch_size, 1), cur_pos, dtype=jnp.int32)
        cache_position = jnp.array([cur_pos], dtype=jnp.int32)
        mask = make_decode_mask(cur_pos, max_cache_len, dtype=model.dtype)

        logits, kv_caches = model(
            next_token, positions, mask,
            kv_caches=kv_caches, cache_position=cache_position,
        )

        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        generated.append(next_token)

        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                hit = jnp.all(next_token == eos_token_id)
            else:
                hit = jnp.all(jnp.isin(next_token, jnp.array(eos_token_id)))
            if hit:
                break

    return jnp.concatenate([input_ids] + generated, axis=1)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["temperature", "top_k", "top_p"])
def _jit_sample(
    logits: jax.Array,
    presence: jax.Array,
    rng_key: jax.Array,
    repetition_penalty: float,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled sampling: penalty → temperature → top_k → top_p → categorical.

    Top-p is applied AFTER top-k on the reduced candidate set for efficiency.
    This avoids sorting the full vocab (155K) — only sorts top_k candidates.
    """
    batch_size = logits.shape[0]

    # 1. Repetition penalty
    penalized = jnp.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
    logits = jnp.where(presence, penalized, logits)

    # 2. Temperature
    if temperature != 1.0:
        logits = logits / temperature

    # 3. Top-k: reduce to k candidates (default 200 when only top_p is used)
    effective_k = top_k if top_k > 0 else 200
    top_k_vals, top_k_idx = jax.lax.top_k(logits, effective_k)  # [B, k]

    # 4. Top-p on the reduced set (sort k elements instead of 155K)
    if top_p < 1.0:
        sorted_order = jnp.argsort(-top_k_vals, axis=-1)
        sorted_vals = jnp.take_along_axis(top_k_vals, sorted_order, axis=-1)
        probs = jax.nn.softmax(sorted_vals.astype(jnp.float32), axis=-1)
        cum_probs = jnp.cumsum(probs, axis=-1)
        mask = (cum_probs - probs) >= top_p
        sorted_vals = jnp.where(mask, -1e9, sorted_vals)
        unsort_order = jnp.argsort(sorted_order, axis=-1)
        top_k_vals = jnp.take_along_axis(sorted_vals, unsort_order, axis=-1)

    # 5. Sample from the k candidates
    rng_key, sample_key = jax.random.split(rng_key)
    selected = jax.random.categorical(sample_key, top_k_vals, axis=-1)
    token = top_k_idx[jnp.arange(batch_size), selected]
    return token[:, None], rng_key


# ---------------------------------------------------------------------------
# Sampling generation (with KV Cache)
# ---------------------------------------------------------------------------

def sample_generate(
    model,
    input_ids: jax.Array,
    max_new_tokens: int = 128,
    max_cache_len: int | None = None,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    eos_token_id: int | list[int] | None = None,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """Sampling generation with KV Cache, temperature, top-k, top-p, and repetition penalty.

    Sampling logic is JIT-compiled separately from the model forward pass.
    Repetition penalty presence mask is maintained on the host (numpy) side
    to avoid dynamic shapes inside JIT.

    Args:
        model: GLM4ForCausalLM instance.
        input_ids: [batch, prompt_len]
        max_new_tokens: Max new tokens.
        max_cache_len: Pre-allocated cache length.
        temperature: Sampling temperature (>0). Lower = more deterministic.
        top_k: Top-k filtering. 0 = disabled.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        repetition_penalty: Penalty for repeated tokens. 1.0 = disabled, >1.0 = penalize.
        eos_token_id: Stop token(s).
        rng_key: JAX random key for sampling.

    Returns:
        generated_ids: [batch, prompt_len + generated_len]
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    batch_size, prompt_len = input_ids.shape
    if max_cache_len is None:
        max_cache_len = prompt_len + max_new_tokens

    # Repetition penalty: maintain presence mask on device [B, vocab_size]
    vocab_size = model.config.vocab_size
    presence = jnp.zeros((batch_size, vocab_size), dtype=jnp.float32)
    # Mark prompt tokens as present
    batch_idx = jnp.arange(batch_size)[:, None]
    presence = presence.at[batch_idx, input_ids].set(1.0)

    # 1. Init KV Cache
    kv_caches = _init_caches(model, batch_size, max_cache_len)

    # 2. Prefill
    positions = make_positions(prompt_len, batch_size)
    cache_position = jnp.arange(prompt_len, dtype=jnp.int32)
    mask = make_prefill_mask(prompt_len, max_cache_len, dtype=model.dtype)

    logits, kv_caches = model(
        input_ids, positions, mask,
        kv_caches=kv_caches, cache_position=cache_position,
    )

    # Sample first token
    next_token, rng_key = _jit_sample(
        logits[:, -1, :], presence, rng_key,
        repetition_penalty, temperature, top_k, top_p,
    )
    generated = [next_token]
    presence = presence.at[jnp.arange(batch_size), next_token[:, 0]].set(1.0)

    # 3. Decode
    for step in range(max_new_tokens - 1):
        cur_pos = prompt_len + step

        positions = jnp.full((batch_size, 1), cur_pos, dtype=jnp.int32)
        cache_position = jnp.array([cur_pos], dtype=jnp.int32)
        mask = make_decode_mask(cur_pos, max_cache_len, dtype=model.dtype)

        logits, kv_caches = model(
            next_token, positions, mask,
            kv_caches=kv_caches, cache_position=cache_position,
        )

        next_token, rng_key = _jit_sample(
            logits[:, -1, :], presence, rng_key,
            repetition_penalty, temperature, top_k, top_p,
        )
        generated.append(next_token)
        presence = presence.at[jnp.arange(batch_size), next_token[:, 0]].set(1.0)

        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                hit = jnp.all(next_token == eos_token_id)
            else:
                hit = jnp.all(jnp.isin(next_token, jnp.array(eos_token_id)))
            if hit:
                break

    return jnp.concatenate([input_ids] + generated, axis=1)


# ---------------------------------------------------------------------------
# No-cache generation (for correctness validation only)
# ---------------------------------------------------------------------------

def greedy_generate_no_cache(
    model,
    input_ids: jax.Array,
    max_new_tokens: int = 128,
    eos_token_id: int | list[int] | None = None,
) -> jax.Array:
    """Greedy generation WITHOUT KV Cache (for validation).

    Each step does a full forward pass over the entire sequence.
    Use this to verify that the cached version produces identical results.
    """
    current_ids = input_ids

    for _ in range(max_new_tokens):
        seq_len = current_ids.shape[1]
        positions = make_positions(seq_len, current_ids.shape[0])
        mask = make_causal_mask(seq_len, dtype=model.dtype)

        logits, _ = model(current_ids, positions, mask)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)

        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                hit = jnp.all(next_token == eos_token_id)
            else:
                hit = jnp.all(jnp.isin(next_token, jnp.array(eos_token_id)))
            if hit:
                break

    return current_ids
