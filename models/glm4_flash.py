"""GLM-4.7-Flash MoE + MLA model implementation in JAX/Flax NNX.

Architecture: Embedding -> N x DecoderLayer(RMSNorm -> MLA -> RMSNorm -> MLP/MoE) -> RMSNorm -> LMHead

Key differences from GLM-4-9B Dense:
  - MLA attention (compressed Q/KV) instead of GQA
  - MoE layers (64 experts + 1 shared) for layers >= first_k_dense_replace
  - Dense MLP for layer 0 (first_k_dense_replace=1)
  - No attention bias
  - head_dim = qk_nope_head_dim + qk_rope_head_dim = 256

Reference: zai-org/GLM-4.7-Flash
"""

import jax
import jax.numpy as jnp
from flax import nnx

from configs.model_config import ModelConfig
from layers.embedding import Embed, ParallelLMHead
from layers.kv_cache import KVCache
from layers.mla_attention import MLAAttention
from layers.moe import DenseMLP, MoELayer
from layers.normalization import RMSNorm


class GLM4FlashDecoderLayer(nnx.Module):
    """Single decoder layer for GLM-4.7-Flash.

    Uses MLA attention for all layers.
    MLP is either dense (layer 0) or MoE (layers 1+).
    """

    def __init__(
        self,
        config: ModelConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id

        self.self_attn = MLAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            mesh=mesh,
            dtype=dtype,
        )

        # Dense MLP for first layers, MoE for the rest
        if layer_id < config.first_k_dense_replace:
            self.mlp = DenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
            )
        else:
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                shared_intermediate_size=config.intermediate_size,
                num_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_topk_prob=config.norm_topk_prob,
                mesh=mesh,
                dtype=dtype,
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: KVCache | None = None,
        cache_position: jax.Array | None = None,
    ) -> tuple[jax.Array, KVCache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_cache = self.self_attn(
            hidden_states, positions, attention_mask,
            kv_cache=kv_cache, cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kv_cache


class GLM4FlashModel(nnx.Module):
    """GLM-4.7-Flash transformer backbone."""

    def __init__(
        self,
        config: ModelConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.List([
            GLM4FlashDecoderLayer(config, mesh, layer_id=i, dtype=dtype)
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_caches: list[KVCache] | None = None,
        cache_position: jax.Array | None = None,
    ) -> tuple[jax.Array, list[KVCache] | None]:
        hidden_states = self.embed_tokens(input_ids)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, layer_cache = layer(
                hidden_states, positions, attention_mask,
                kv_cache=layer_cache, cache_position=cache_position,
            )
            new_caches.append(layer_cache)

        hidden_states = self.norm(hidden_states)
        out_caches = new_caches if kv_caches is not None else None
        return hidden_states, out_caches


class GLM4FlashForCausalLM(nnx.Module):
    """GLM-4.7-Flash for causal language modeling."""

    def __init__(
        self,
        config: ModelConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype

        self.model = GLM4FlashModel(config, mesh, dtype=dtype)

        if not config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                dtype=dtype,
                mesh=mesh,
            )
        else:
            self.lm_head = None

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_caches: list[KVCache] | None = None,
        cache_position: jax.Array | None = None,
    ) -> tuple[jax.Array, list[KVCache] | None]:
        hidden_states, kv_caches = self.model(
            input_ids, positions, attention_mask,
            kv_caches=kv_caches, cache_position=cache_position,
        )

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = jnp.einsum(
                '...h,vh->...v',
                hidden_states.astype(self.dtype),
                self.model.embed_tokens.embedding.value,
            )
        return logits, kv_caches

    def load_weights(self, model_config: ModelConfig):
        """Load weights from HuggingFace safetensors.

        Non-expert weights are loaded normally. Expert weights are loaded
        individually, stacked into [E, H, I] tensors, then assigned to the
        model's stacked expert params.
        """
        from utils.weight_utils import WeightLoader, WeightMapping

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        # Split mappings: base (non-expert) vs expert
        all_mappings = _create_glm4_flash_weight_mappings(model_config)
        base_mappings = [m for m in all_mappings if not m.hf_key.startswith("__expert__")]
        expert_specs = [m for m in all_mappings if m.hf_key.startswith("__expert__")]

        # Load base weights normally
        loader.load_weights(base_mappings)

        # Load and stack expert weights
        _load_stacked_experts(self, loader, model_config)


def _load_stacked_experts(model, loader, config: ModelConfig):
    """Load individual expert weights from safetensors, stack into [E, H, I]."""
    import glob
    import logging
    import os

    import numpy as np
    from safetensors import safe_open

    logger = logging.getLogger(__name__)
    model_path = config.model_path
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    # Build index
    key_to_file: dict[str, str] = {}
    for st_file in safetensor_files:
        with safe_open(st_file, framework="numpy", device="cpu") as f:
            for key in f.keys():
                key_to_file[key] = st_file

    cpu = jax.devices("cpu")[0]
    tensor_cache: dict[str, np.ndarray] = {}

    for i in range(config.num_hidden_layers):
        if i < config.first_k_dense_replace:
            continue

        layer = model.model.layers[i]

        for proj_name, param_name in [
            ("gate_proj", "expert_gate_weight"),
            ("up_proj", "expert_up_weight"),
            ("down_proj", "expert_down_weight"),
        ]:
            expert_weights = []
            for e in range(config.n_routed_experts):
                hf_key = f"model.layers.{i}.mlp.experts.{e}.{proj_name}.weight"
                if hf_key not in key_to_file:
                    raise KeyError(f"Missing expert weight: {hf_key}")

                # Read with caching (same file may contain multiple experts)
                file_path = key_to_file[hf_key]
                if file_path not in tensor_cache:
                    tensor_cache.clear()  # only cache one file at a time
                with safe_open(file_path, framework="numpy", device="cpu") as f:
                    w = f.get_tensor(hf_key)

                # HF format [out, in] → JAX [in, out]
                w = w.T
                expert_weights.append(w)

            # Stack: [E, in, out]
            stacked = np.stack(expert_weights, axis=0)

            # Convert to bf16 and assign on CPU
            param = getattr(layer.mlp, param_name)
            param.value = jax.device_put(
                jnp.asarray(stacked, dtype=jnp.bfloat16), cpu,
            )

        logger.info("Stacked expert weights for layer %d", i)


def _create_glm4_flash_weight_mappings(config: ModelConfig) -> list:
    """Build HF weight name -> JAX model path mapping for GLM-4.7-Flash.

    Expert weights (model.layers.*.mlp.experts.*) are NOT included here;
    they are loaded separately by _load_stacked_experts().
    """
    from utils.weight_utils import WeightMapping

    mappings = [
        # Embeddings
        WeightMapping(
            hf_key="model.embed_tokens.weight",
            target_path="model.embed_tokens.embedding",
            transpose=False,
        ),
        # Final norm
        WeightMapping(
            hf_key="model.norm.weight",
            target_path="model.norm.scale",
            transpose=False,
        ),
    ]

    # LM Head
    if not config.tie_word_embeddings:
        mappings.append(WeightMapping(
            hf_key="lm_head.weight",
            target_path="lm_head.embedding",
            transpose=False,
        ))

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"

        # Layer norms
        mappings.append(WeightMapping(
            hf_key=f"{p}.input_layernorm.weight",
            target_path=f"{p}.input_layernorm.scale",
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.post_attention_layernorm.weight",
            target_path=f"{p}.post_attention_layernorm.scale",
        ))

        # --- MLA Attention ---
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.q_a_proj.weight",
            target_path=f"{p}.self_attn.q_a_proj.weight",
            transpose=True,
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.q_a_layernorm.weight",
            target_path=f"{p}.self_attn.q_a_layernorm.scale",
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.q_b_proj.weight",
            target_path=f"{p}.self_attn.q_b_proj.weight",
            transpose=True,
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.kv_a_proj_with_mqa.weight",
            target_path=f"{p}.self_attn.kv_a_proj_with_mqa.weight",
            transpose=True,
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.kv_a_layernorm.weight",
            target_path=f"{p}.self_attn.kv_a_layernorm.scale",
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.kv_b_proj.weight",
            target_path=f"{p}.self_attn.kv_b_proj.weight",
            transpose=True,
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.o_proj.weight",
            target_path=f"{p}.self_attn.o_proj.weight",
            transpose=True,
        ))

        # --- MLP ---
        if i < config.first_k_dense_replace:
            # Dense MLP
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                mappings.append(WeightMapping(
                    hf_key=f"{p}.mlp.{proj}.weight",
                    target_path=f"{p}.mlp.{proj}.weight",
                    transpose=True,
                ))
        else:
            # MoE: gate (router)
            mappings.append(WeightMapping(
                hf_key=f"{p}.mlp.gate.weight",
                target_path=f"{p}.mlp.gate.weight.weight",
                transpose=True,
            ))
            mappings.append(WeightMapping(
                hf_key=f"{p}.mlp.gate.e_score_correction_bias",
                target_path=f"{p}.mlp.gate.e_score_correction_bias",
            ))

            # MoE: routed experts — loaded separately by _load_stacked_experts()
            # (not included in base mappings)

            # MoE: shared expert
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                mappings.append(WeightMapping(
                    hf_key=f"{p}.mlp.shared_experts.{proj}.weight",
                    target_path=f"{p}.mlp.shared_experts.{proj}.weight",
                    transpose=True,
                ))

    return mappings
