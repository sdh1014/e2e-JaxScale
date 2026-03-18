"""Mixture of Experts (MoE) layer for GLM-4.7-Flash.

GLM-4.7-Flash MoE config:
  - 64 routed experts, 1 shared expert
  - top-4 routing (num_experts_per_tok=4)
  - moe_intermediate_size=1536 per routed expert
  - intermediate_size=10240 for shared expert (and dense layers)
  - routed_scaling_factor=1.8
  - first_k_dense_replace=1 (layer 0 uses dense MLP)

EP implementation: expert weights are stacked into [E, H, I] tensors and
sharded along the E dimension across EP devices. Uses shard_map with
local gather + psum for efficient EP communication:
  - Each EP device only gathers its local experts (K=4 gathers, not E=32 einsum)
  - Single psum("expert") per layer for cross-EP all-reduce
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from layers.linear import LinearBase


class MoEGate(nnx.Module):
    """Router for MoE: computes expert scores and selects top-k."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        self.weight = LinearBase(
            hidden_size, num_experts, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, None),  # replicated (small)
        )
        self.e_score_correction_bias = nnx.Param(
            jnp.zeros((), dtype=jnp.float32),  # placeholder, loaded from weights
        )

    def __call__(self, hidden_states: jax.Array):
        """Compute routing weights.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            top_k_weights: [batch, seq_len, k] normalized expert weights
            top_k_indices: [batch, seq_len, k] expert indices
        """
        logits = self.weight(hidden_states)  # [B, S, E]
        logits = logits.astype(jnp.float32)
        scores = jax.nn.softmax(logits, axis=-1)

        # Apply correction bias and select top-k
        corrected = scores + self.e_score_correction_bias.value
        top_k_weights, top_k_indices = jax.lax.top_k(corrected, self.num_experts_per_tok)

        # Use original scores (not corrected) for the actual weights
        top_k_weights = jnp.take_along_axis(scores, top_k_indices, axis=-1)

        if self.norm_topk_prob:
            top_k_weights = top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-20)

        top_k_weights = top_k_weights * self.routed_scaling_factor
        return top_k_weights.astype(hidden_states.dtype), top_k_indices


class SharedExpertMLP(nnx.Module):
    """Shared expert SwiGLU MLP (larger, always active for all tokens).

    Uses TP sharding like the dense model's MLP since intermediate_size
    is large (10240).
    """

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
            kernel_axes=(None, "tensor"),  # column-parallel
        )
        self.up_proj = LinearBase(
            hidden_size, intermediate_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.down_proj = LinearBase(
            intermediate_size, hidden_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None),  # row-parallel
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nnx.Module):
    """MoE layer with stacked expert weights for EP sharding.

    Expert weights stored as [E, H, I] tensors, sharded by E for EP.
    Uses shard_map with local gather + psum("expert") for efficient routing:
    each EP device gathers only from its local experts (K=4 gathers),
    then a single psum combines across EP devices.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.mesh = mesh

        self.gate = MoEGate(
            hidden_size, num_experts, num_experts_per_tok,
            routed_scaling_factor, norm_topk_prob, mesh, dtype,
        )

        # Stacked expert weights: [E, H, I] and [E, I, H]
        # Scalar placeholders — filled by load_and_stack_experts()
        self.expert_gate_weight = nnx.Param(jnp.zeros((), dtype=dtype))
        self.expert_up_weight = nnx.Param(jnp.zeros((), dtype=dtype))
        self.expert_down_weight = nnx.Param(jnp.zeros((), dtype=dtype))

        self.shared_experts = SharedExpertMLP(
            hidden_size, shared_intermediate_size, mesh, dtype,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """Forward pass with top-k local gather routing.

        With EP>1: shard_map ensures each device gathers only from local experts.
        Without EP: simple local gather loop over K=4 selected experts.
        """
        B, S, H = hidden_states.shape

        # Route
        top_k_weights, top_k_indices = self.gate(hidden_states)  # [B,S,K], [B,S,K]

        hidden_flat = hidden_states.reshape(-1, H)  # [N, H]
        tk_w = top_k_weights.reshape(-1, self.num_experts_per_tok)  # [N, K]
        tk_i = top_k_indices.reshape(-1, self.num_experts_per_tok)  # [N, K]

        ep_size = self.mesh.shape.get("expert", 1)

        if ep_size > 1:
            routed_output = self._forward_ep(
                hidden_flat, tk_w, tk_i,
                self.expert_gate_weight.value,
                self.expert_up_weight.value,
                self.expert_down_weight.value,
                ep_size,
            )
        else:
            routed_output = _topk_local_forward(
                hidden_flat, tk_w, tk_i,
                self.expert_gate_weight.value,
                self.expert_up_weight.value,
                self.expert_down_weight.value,
                self.num_experts_per_tok,
            )

        routed_output = routed_output.reshape(B, S, H)
        shared_output = self.shared_experts(hidden_states)
        return routed_output + shared_output

    def _forward_ep(self, hidden_flat, tk_w, tk_i, gate_w, up_w, down_w, ep_size):
        """EP forward using shard_map: local gather + psum."""
        num_experts = self.num_experts
        num_experts_per_tok = self.num_experts_per_tok
        experts_per_device = num_experts // ep_size

        def _ep_fn(hidden_flat, tk_w, tk_i, gate_w, up_w, down_w):
            expert_offset = jax.lax.axis_index("expert") * experts_per_device
            N, H = hidden_flat.shape
            output = jnp.zeros((N, H), dtype=hidden_flat.dtype)

            for k in range(num_experts_per_tok):
                expert_ids = tk_i[:, k]       # [N] global expert IDs
                weights_k = tk_w[:, k]        # [N] routing weights

                # Local expert index; clip for safe gather
                local_ids = expert_ids - expert_offset
                is_local = (local_ids >= 0) & (local_ids < experts_per_device)
                safe_ids = jnp.clip(local_ids, 0, experts_per_device - 1)

                # Local gather — no cross-device communication
                g_w = gate_w[safe_ids]  # [N, H, I]
                u_w = up_w[safe_ids]
                d_w = down_w[safe_ids]  # [N, I, H]

                # SwiGLU
                gate_out = jnp.einsum('nh,nhi->ni', hidden_flat, g_w)
                up_out = jnp.einsum('nh,nhi->ni', hidden_flat, u_w)
                activated = jax.nn.silu(gate_out) * up_out
                expert_out = jnp.einsum('ni,nih->nh', activated, d_w)

                # Only the EP device owning this expert contributes
                mask = (weights_k * is_local.astype(weights_k.dtype))[:, None]
                output = output + expert_out * mask

            # Single all-reduce across EP devices
            output = jax.lax.psum(output, "expert")
            return output

        in_specs = (
            P(None, None), P(None, None), P(None, None),
            P("expert", None, None), P("expert", None, None), P("expert", None, None),
        )
        out_specs = P(None, None)

        return shard_map(
            _ep_fn,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )(hidden_flat, tk_w, tk_i, gate_w, up_w, down_w)


def _topk_local_forward(hidden_flat, tk_w, tk_i, gate_w, up_w, down_w, K):
    """Top-k forward without EP: all experts are local."""
    N, H = hidden_flat.shape
    output = jnp.zeros((N, H), dtype=hidden_flat.dtype)

    for k in range(K):
        expert_ids = tk_i[:, k]
        weights_k = tk_w[:, k]

        g_w = gate_w[expert_ids]
        u_w = up_w[expert_ids]
        d_w = down_w[expert_ids]

        gate_out = jnp.einsum('nh,nhi->ni', hidden_flat, g_w)
        up_out = jnp.einsum('nh,nhi->ni', hidden_flat, u_w)
        activated = jax.nn.silu(gate_out) * up_out
        expert_out = jnp.einsum('ni,nih->nh', activated, d_w)

        output = output + expert_out * weights_k[:, None]

    return output


class DenseMLP(nnx.Module):
    """Dense SwiGLU MLP for the first_k_dense_replace layers.

    Uses TP sharding (same as GLM-4-9B MLP).
    """

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
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
