"""Linear layer with sharding support for GLM-4 models."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


class LinearBase(nnx.Module):
    """Linear layer with optional sharding via kernel_axes.

    Weight shape: (input_size, output_size)
    Forward: output = x @ weight + bias

    Args:
        input_size: Input dimension.
        output_size: Output dimension.
        mesh: JAX device mesh for sharding.
        use_bias: Whether to include bias.
        params_dtype: Data type for parameters.
        kernel_axes: Tuple of axis names for weight sharding.
            e.g. (None, "tensor") for column-parallel,
                 ("tensor", None) for row-parallel.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        mesh: jax.sharding.Mesh,
        use_bias: bool = False,
        params_dtype: jnp.dtype = jnp.bfloat16,
        kernel_axes: Sequence[str | None] | None = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.mesh = mesh
        self.params_dtype = params_dtype
        self.kernel_axes = kernel_axes or (None, None)

        self.weight = nnx.Param(
            jnp.zeros((input_size, output_size), dtype=params_dtype),
        )
        if use_bias:
            self.bias = nnx.Param(
                jnp.zeros((output_size,), dtype=params_dtype),
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass. Returns output tensor."""
        output = lax.dot_general(
            x,
            self.weight.value,
            (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=self.params_dtype,
        )
        if self.bias is not None:
            output = output + self.bias.value
        return output
