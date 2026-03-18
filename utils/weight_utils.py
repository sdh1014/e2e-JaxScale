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

        Weights are placed on CPU during loading to avoid OOM on device.
        shard_model_params() later moves them to the correct devices.
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

        # Convert numpy to JAX array on CPU to avoid device OOM during loading
        cpu = jax.devices("cpu")[0]
        jax_value = jax.device_put(np.asarray(value_np, dtype=jnp.dtype(self.dtype)), cpu)

        if isinstance(current, nnx.Param):
            current.value = jnp.zeros((), dtype=jnp.float32)  # free old
            current.value = jax_value
        else:
            setattr(obj, final_attr, nnx.Param(jax_value))


def shard_model_params(model, mesh: jax.sharding.Mesh):
    """Apply TP sharding to all model parameters and move to devices.

    Must be called after load_weights(). Walks the model tree recursively
    and uses jax.device_put to distribute each LinearBase weight (and bias)
    according to its kernel_axes PartitionSpec. Non-sharded params are replicated.
    Also handles moving weights from CPU to device for tp=1.

    Works for both GLM-4-9B (Dense) and GLM-4.7-Flash (MoE + MLA).

    Args:
        model: Model instance with weights already loaded (possibly on CPU).
        mesh: JAX device mesh.
    """
    from layers.linear import LinearBase
    from layers.normalization import RMSNorm

    tp_size = mesh.shape.get("tensor", 1)

    sharded_count = [0]  # mutable counter for nested function
    replicated_2d = NamedSharding(mesh, P(None, None))
    replicated_1d = NamedSharding(mesh, P(None))

    def _walk_and_shard(module):
        """Recursively walk module tree and apply sharding."""
        if isinstance(module, LinearBase):
            _shard_linear(module, mesh)
            sharded_count[0] += 1
            return
        if isinstance(module, RMSNorm):
            module.scale.value = jax.device_put(module.scale.value, replicated_1d)
            return

        # Handle MoE stacked expert weights: P("expert", None, None)
        from layers.moe import MoELayer
        if isinstance(module, MoELayer):
            ep_size = mesh.shape.get("expert", 1)
            if ep_size > 1:
                ep_pspec = P("expert", None, None)
            else:
                ep_pspec = P(None, None, None)
            ep_sharding = NamedSharding(mesh, ep_pspec)

            for attr in ("expert_gate_weight", "expert_up_weight", "expert_down_weight"):
                param = getattr(module, attr)
                if param.value.ndim == 3:  # [E, H, I] — loaded
                    param.value = jax.device_put(param.value, ep_sharding)
                    sharded_count[0] += 1

            # Continue to recurse into gate, shared_experts, etc.

        # Recurse into children
        for name in vars(module):
            child = getattr(module, name, None)
            if child is None:
                continue
            if isinstance(child, nnx.Param):
                # Move any remaining CPU params to device (replicated)
                if hasattr(child.value, 'devices') and not any(
                    d.platform != 'cpu' for d in child.value.devices()
                ):
                    child.value = jax.device_put(child.value,
                                                 NamedSharding(mesh, P(*([None] * child.value.ndim))))
            elif isinstance(child, (nnx.Module, LinearBase, RMSNorm)):
                _walk_and_shard(child)
            elif isinstance(child, (list, nnx.List)):
                for item in child:
                    if isinstance(item, nnx.Module):
                        _walk_and_shard(item)

    _walk_and_shard(model)

    # Embedding and LMHead: replicate
    model.model.embed_tokens.embedding.value = jax.device_put(
        model.model.embed_tokens.embedding.value, replicated_2d
    )
    if model.lm_head is not None:
        model.lm_head.embedding.value = jax.device_put(
            model.lm_head.embedding.value, replicated_2d
        )

    logger.info("Sharded %d linear layers across TP=%d devices", sharded_count[0], tp_size)


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
