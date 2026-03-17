"""Embedding and LM Head layers for GLM-4 models."""

import jax
import jax.numpy as jnp
from flax import nnx


class Embed(nnx.Module):
    """Token embedding layer.

    Maps integer token IDs to dense vectors.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype

        self.embedding = nnx.Param(
            jnp.zeros((num_embeddings, features), dtype=dtype),
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """Look up embeddings for input token IDs.

        Args:
            input_ids: Integer tensor of token IDs, any shape.

        Returns:
            Embedded tensor with shape (*input_ids.shape, features).
        """
        return jnp.take(self.embedding.value, input_ids, axis=0)


class ParallelLMHead(nnx.Module):
    """Language model head: projects hidden states to vocabulary logits.

    Weight shape: (vocab_size, hidden_size) — same as embedding but
    used as a linear projection: logits = hidden @ weight.T
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype = dtype

        # Weight stored as (vocab_size, hidden_size) like HF convention
        self.embedding = nnx.Param(
            jnp.zeros((vocab_size, hidden_size), dtype=dtype),
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """Compute logits from hidden states.

        Args:
            hidden_states: [..., hidden_size]

        Returns:
            logits: [..., vocab_size]
        """
        # Use einsum to avoid materializing the transpose of the large embedding
        return jnp.einsum(
            '...h,vh->...v',
            hidden_states.astype(self.dtype),
            self.embedding.value,
        )

    def tie_weights(self, embed_tokens: Embed):
        """Share weights with the input embedding layer."""
        self.embedding = embed_tokens.embedding
