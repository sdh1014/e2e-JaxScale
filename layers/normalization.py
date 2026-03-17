"""RMSNorm implementation for GLM-4 models."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import PartitionSpec as P


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    Used in GLM-4 for both input_layernorm and post_attention_layernorm.
    Computes: x * rsqrt(mean(x^2) + eps) * scale
    """

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.param_dtype = param_dtype

        self.scale = nnx.Param(
            jnp.ones((hidden_size,), dtype=param_dtype),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x_f32 = jnp.asarray(x, jnp.float32)
        variance = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
        x_normed = x_f32 * lax.rsqrt(variance + self.epsilon)
        output = x_normed * jnp.asarray(self.scale.value, jnp.float32)
        return jnp.asarray(output, orig_dtype)
