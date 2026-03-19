"""Model configuration for GLM-4 series models.

Parses HuggingFace config.json into a unified dataclass.
Supports both GLM-4-9B (Dense) and GLM-4.7-Flash (MoE + MLA).
"""

import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Unified config for GLM-4 Dense and GLM-4.7-Flash MoE models."""

    # --- Core architecture ---
    hidden_size: int = 4096
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int | None = None  # inferred if None
    intermediate_size: int = 13696
    vocab_size: int = 151552
    max_position_embeddings: int = 131072

    # --- Normalization / Activation ---
    rms_norm_eps: float = 1.5625e-07
    hidden_act: str = "silu"

    # --- Attention ---
    attention_bias: bool = True
    rope_theta: float = 10000.0

    # --- Embeddings ---
    tie_word_embeddings: bool = False

    # --- MoE (optional, for GLM-4.7-Flash) ---
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    routed_scaling_factor: float = 1.0
    topk_method: str = "greedy"
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 0

    # --- MLA (optional, for GLM-4.7-Flash) ---
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None

    # --- Runtime ---
    model_path: str = ""
    model_type: str = "glm"
    eos_token_id: int | list[int] | None = None

    def __post_init__(self):
        if self.head_dim is None:
            if self.is_mla:
                # MLA: head_dim = qk_nope + qk_rope (for K), v_head_dim (for V)
                self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            else:
                self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts is not None

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load config from a HuggingFace model directory."""
        import os

        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw = json.load(f)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for key, value in raw.items():
            if key in known_fields:
                kwargs[key] = value

        # Load EOS token IDs from generation_config.json if available
        gen_config_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(gen_config_path):
            with open(gen_config_path) as f:
                gen_raw = json.load(f)
            if "eos_token_id" in gen_raw:
                kwargs["eos_token_id"] = gen_raw["eos_token_id"]

        kwargs["model_path"] = model_path
        return cls(**kwargs)
