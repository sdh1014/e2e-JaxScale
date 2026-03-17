"""GLM-4-9B Dense model implementation in JAX/Flax NNX.

Architecture: Embedding -> N x DecoderLayer(RMSNorm -> Attention -> RMSNorm -> SwiGLU MLP) -> RMSNorm -> LMHead

Reference: THUDM/glm-4-9b-chat-hf
"""

import jax
import jax.numpy as jnp
from flax import nnx

from configs.model_config import ModelConfig
from layers.attention import GQAAttention
from layers.embedding import Embed, ParallelLMHead
from layers.linear import LinearBase
from layers.normalization import RMSNorm


class GLM4MLP(nnx.Module):
    """SwiGLU MLP: gate_proj * silu(up_proj) -> down_proj."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.gate_proj = LinearBase(
            hidden_size, intermediate_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.up_proj = LinearBase(
            hidden_size, intermediate_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.down_proj = LinearBase(
            intermediate_size, hidden_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(jax.nn.silu(gate) * up)


class GLM4DecoderLayer(nnx.Module):
    """Single transformer decoder layer.

    Flow: RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
    """

    def __init__(
        self,
        config: ModelConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id

        self.self_attn = GQAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            mesh=mesh,
            attention_bias=config.attention_bias,
            dtype=dtype,
        )

        self.mlp = GLM4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
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
    ) -> jax.Array:
        """Forward pass with pre-norm and residual connections.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            positions: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len]

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GLM4Model(nnx.Module):
    """GLM-4 transformer backbone: Embedding -> Layers -> Final Norm."""

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
        )

        self.layers = nnx.List([
            GLM4DecoderLayer(config, mesh, layer_id=i, dtype=dtype)
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
    ) -> jax.Array:
        """Forward pass through the full model backbone.

        Args:
            input_ids: [batch, seq_len] token IDs.
            positions: [batch, seq_len] position indices.
            attention_mask: [batch, 1, seq_len, seq_len] causal mask.

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GLM4ForCausalLM(nnx.Module):
    """GLM-4-9B for causal language modeling.

    Full model: GLM4Model + LMHead
    """

    def __init__(
        self,
        config: ModelConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype

        self.model = GLM4Model(config, mesh, dtype=dtype)

        if not config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                dtype=dtype,
            )
        else:
            self.lm_head = None

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass returning logits.

        Args:
            input_ids: [batch, seq_len]
            positions: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.model(input_ids, positions, attention_mask)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings: use embed_tokens weight as LM head
            logits = jnp.einsum(
                '...h,vh->...v',
                hidden_states.astype(self.dtype),
                self.model.embed_tokens.embedding.value,
            )
        return logits

    def load_weights(self, model_config: ModelConfig):
        """Load weights from HuggingFace safetensors."""
        from utils.weight_utils import WeightLoader

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        mappings = _create_glm4_weight_mappings(model_config)
        loader.load_weights(mappings)


def _create_glm4_weight_mappings(config: ModelConfig) -> list["WeightMapping"]:
    """Build HF weight name -> JAX model path mapping for GLM-4-9B."""
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

    # Per-layer mappings
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

        # Attention projections: q/k/v have weight + bias, o has weight only
        for proj in ["q_proj", "k_proj", "v_proj"]:
            mappings.append(WeightMapping(
                hf_key=f"{p}.self_attn.{proj}.weight",
                target_path=f"{p}.self_attn.{proj}.weight",
                transpose=True,
            ))
            if config.attention_bias:
                mappings.append(WeightMapping(
                    hf_key=f"{p}.self_attn.{proj}.bias",
                    target_path=f"{p}.self_attn.{proj}.bias",
                ))

        # o_proj: weight only (no bias in GLM-4-9B HF checkpoint)
        mappings.append(WeightMapping(
            hf_key=f"{p}.self_attn.o_proj.weight",
            target_path=f"{p}.self_attn.o_proj.weight",
            transpose=True,
        ))

        # MLP: gate_up_proj is fused in HF, split into gate_proj and up_proj
        mappings.append(WeightMapping(
            hf_key=f"{p}.mlp.gate_up_proj.weight",
            target_path=f"{p}.mlp.gate_proj.weight",
            transpose=True,
            split_dim=0, split_count=2, split_index=0,
        ))
        mappings.append(WeightMapping(
            hf_key=f"{p}.mlp.gate_up_proj.weight",
            target_path=f"{p}.mlp.up_proj.weight",
            transpose=True,
            split_dim=0, split_count=2, split_index=1,
        ))
        # down_proj
        mappings.append(WeightMapping(
            hf_key=f"{p}.mlp.down_proj.weight",
            target_path=f"{p}.mlp.down_proj.weight",
            transpose=True,
        ))

    return mappings
