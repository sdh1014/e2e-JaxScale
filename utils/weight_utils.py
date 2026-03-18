"""Weight loading utilities for HuggingFace safetensors -> JAX model."""

import glob
import logging
import os
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors import safe_open

from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class WeightMapping:
    """Describes how to map one HF weight to the JAX model.

    Attributes:
        hf_key: The key in the HF safetensors file.
        target_path: Dot-separated path to the target parameter in the JAX model.
        transpose: Whether to transpose the weight (HF: [out, in] -> JAX: [in, out]).
        split_dim: Dimension to split along (for fused weights like gate_up_proj).
        split_count: Number of chunks to split into.
        split_index: Which chunk to take (0-indexed).
    """
    hf_key: str
    target_path: str
    transpose: bool = False
    split_dim: int | None = None
    split_count: int | None = None
    split_index: int | None = None


class WeightLoader:
    """Loads HuggingFace safetensors weights into a Flax NNX model.

    Usage:
        loader = WeightLoader(model, model_config, mesh, dtype)
        loader.load_weights(weight_mappings)
    """

    def __init__(
        self,
        model: nnx.Module,
        model_config: ModelConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.model = model
        self.model_config = model_config
        self.mesh = mesh
        self.dtype = dtype

    def load_weights(self, mappings: list[WeightMapping]):
        """Load all weights from safetensors files.

        Args:
            mappings: List of WeightMapping descriptors.
        """
        model_path = self.model_config.model_path
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}"
            )

        # Build index: HF key -> file path
        key_to_file: dict[str, str] = {}
        for st_file in safetensor_files:
            with safe_open(st_file, framework="numpy", device="cpu") as f:
                for key in f.keys():
                    key_to_file[key] = st_file

        logger.info(
            "Found %d weights in %d safetensors files",
            len(key_to_file), len(safetensor_files),
        )

        # Cache: avoid re-reading the same tensor (for split weights)
        tensor_cache: dict[str, np.ndarray] = {}

        # Load each mapped weight
        loaded_count = 0
        skipped_keys = []

        for mapping in mappings:
            hf_key = mapping.hf_key
            if hf_key not in key_to_file:
                skipped_keys.append(hf_key)
                continue

            # Read weight (with caching for fused weights)
            if hf_key in tensor_cache:
                weight_np = tensor_cache[hf_key]
            else:
                with safe_open(key_to_file[hf_key], framework="numpy", device="cpu") as f:
                    weight_np = f.get_tensor(hf_key)
                # Cache if this is a split weight (others may need it)
                if mapping.split_dim is not None:
                    tensor_cache[hf_key] = weight_np

            # Split if needed (for fused weights like gate_up_proj)
            if mapping.split_dim is not None:
                chunks = np.array_split(weight_np, mapping.split_count, axis=mapping.split_dim)
                weight_np = chunks[mapping.split_index]
                # Clean cache if this is the last split
                if mapping.split_index == mapping.split_count - 1:
                    tensor_cache.pop(hf_key, None)

            # Transpose if needed (HF: [out, in] -> JAX: [in, out])
            if mapping.transpose:
                weight_np = weight_np.T

            # Set parameter in model (handles dtype conversion and memory management)
            self._set_param(mapping.target_path, weight_np)
            loaded_count += 1

        logger.info("Loaded %d / %d weights", loaded_count, len(mappings))
        if skipped_keys:
            logger.warning("Skipped %d missing weights: %s", len(skipped_keys), skipped_keys[:5])

    def _set_param(self, target_path: str, value_np: np.ndarray):
        """Set a parameter in the model by dot-separated path.

        Handles both nnx.Param attributes and list indexing for layers.
        e.g. "model.layers.3.self_attn.q_proj.weight"

        Frees the old GPU buffer before allocating the new one to
        reduce peak GPU memory during weight loading.
        """
        parts = target_path.split(".")
        obj = self.model

        # Navigate to parent
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)

        # Set the final attribute
        final_attr = parts[-1]
        current = getattr(obj, final_attr)

        if isinstance(current, nnx.Param):
            # Free old GPU buffer first, then convert and assign new value
            current.value = jnp.zeros((), dtype=jnp.float32)
            current.value = jnp.asarray(value_np, dtype=self.dtype)
        else:
            setattr(obj, final_attr, nnx.Param(jnp.asarray(value_np, dtype=self.dtype)))


def shard_model_params(model, mesh: jax.sharding.Mesh):
    """Apply TP sharding to all model parameters based on kernel_axes.

    Must be called after load_weights(). Walks the model tree and uses
    jax.device_put to distribute each LinearBase weight (and bias) according
    to its kernel_axes PartitionSpec. Embedding weights are replicated.

    Args:
        model: GLM4ForCausalLM instance with weights already loaded.
        mesh: JAX device mesh (must have a "tensor" axis for TP).
    """
    from layers.linear import LinearBase

    tp_size = mesh.shape.get("tensor", 1)
    if tp_size <= 1:
        logger.info("TP size is 1, skipping weight sharding")
        return

    sharded_count = 0

    for layer in model.model.layers:
        # Attention projections
        for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj,
                     layer.self_attn.v_proj, layer.self_attn.o_proj]:
            _shard_linear(proj, mesh)
            sharded_count += 1

        # MLP projections
        for proj in [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]:
            _shard_linear(proj, mesh)
            sharded_count += 1

    # Embedding and LMHead: replicate (P(None, None))
    replicated = NamedSharding(mesh, P(None, None))
    model.model.embed_tokens.embedding.value = jax.device_put(
        model.model.embed_tokens.embedding.value, replicated
    )
    if model.lm_head is not None:
        model.lm_head.embedding.value = jax.device_put(
            model.lm_head.embedding.value, replicated
        )

    # RMSNorm scales: replicate (P(None))
    replicated_1d = NamedSharding(mesh, P(None))
    model.model.norm.scale.value = jax.device_put(
        model.model.norm.scale.value, replicated_1d
    )
    for layer in model.model.layers:
        layer.input_layernorm.scale.value = jax.device_put(
            layer.input_layernorm.scale.value, replicated_1d
        )
        layer.post_attention_layernorm.scale.value = jax.device_put(
            layer.post_attention_layernorm.scale.value, replicated_1d
        )

    logger.info("Sharded %d linear layers across TP=%d devices", sharded_count, tp_size)


def _shard_linear(linear, mesh: jax.sharding.Mesh):
    """Apply TP sharding to a single LinearBase module's parameters.

    Weight PartitionSpec comes directly from kernel_axes:
      - Column-parallel (None, "tensor"): output dim sharded
      - Row-parallel ("tensor", None): input dim sharded
    """
    weight_pspec = P(*linear.kernel_axes)
    weight_sharding = NamedSharding(mesh, weight_pspec)
    linear.weight.value = jax.device_put(linear.weight.value, weight_sharding)

    if linear.bias is not None:
        # Bias shape: (output_size,)
        # Column-parallel: bias sharded by "tensor"
        # Row-parallel: bias replicated (kernel_axes[1] is None)
        bias_pspec = P(linear.kernel_axes[1])
        bias_sharding = NamedSharding(mesh, bias_pspec)
        linear.bias.value = jax.device_put(linear.bias.value, bias_sharding)
