"""Profile GLM-4.7-Flash decode step latency breakdown.

Measures time spent in each component per decode step:
- MLA Attention (per layer)
- MoE routed experts (shard_map + local gather)
- Shared expert MLP
- Router (gate)
- Norms + residuals
"""

import argparse
import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from configs.model_config import ModelConfig
from layers.kv_cache import init_kv_caches
from runner import make_prefill_mask, make_decode_mask, make_positions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def profile_components(model, config, mesh, dtype):
    """Profile each component of the model separately."""
    batch_size = 1
    hidden_size = config.hidden_size
    max_cache_len = 64

    # Pick a MoE layer (layer 1) and the dense layer (layer 0)
    moe_layer = model.model.layers[1]
    dense_layer = model.model.layers[0]

    # Dummy inputs
    hidden = jnp.ones((batch_size, 1, hidden_size), dtype=dtype)
    positions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    mask = make_decode_mask(10, max_cache_len, dtype=dtype)

    # Init per-layer KV cache
    if config.is_mla:
        num_kv_heads = config.num_attention_heads
        head_dim = config.head_dim
    else:
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

    k_cache = jnp.zeros((batch_size, max_cache_len, num_kv_heads, head_dim), dtype=dtype)
    v_cache = jnp.zeros((batch_size, max_cache_len, num_kv_heads, head_dim), dtype=dtype)

    if mesh is not None:
        from jax.sharding import NamedSharding, PartitionSpec as P
        kv_sharding = NamedSharding(mesh, P(None, None, "tensor", None))
        k_cache = jax.device_put(k_cache, kv_sharding)
        v_cache = jax.device_put(v_cache, kv_sharding)

    from layers.kv_cache import KVCache
    kv_cache = KVCache(k=k_cache, v=v_cache)
    cache_position = jnp.array([10], dtype=jnp.int32)

    warmup = 3
    repeats = 20

    # --- Profile MLA Attention ---
    @jax.jit
    def run_attention(hidden, positions, mask, kv_cache, cache_position):
        normed = moe_layer.input_layernorm(hidden)
        out, kv = moe_layer.self_attn(normed, positions, mask,
                                       kv_cache=kv_cache, cache_position=cache_position)
        return out, kv

    for _ in range(warmup):
        out, kv_cache_new = run_attention(hidden, positions, mask, kv_cache, cache_position)
        jax.block_until_ready(out)

    t0 = time.time()
    for _ in range(repeats):
        out, _ = run_attention(hidden, positions, mask, kv_cache, cache_position)
        jax.block_until_ready(out)
    attn_ms = (time.time() - t0) / repeats * 1000

    # --- Profile MoE (routed + shared) ---
    @jax.jit
    def run_moe_full(hidden):
        normed = moe_layer.post_attention_layernorm(hidden)
        return moe_layer.mlp(normed)

    for _ in range(warmup):
        out = run_moe_full(hidden)
        jax.block_until_ready(out)

    t0 = time.time()
    for _ in range(repeats):
        out = run_moe_full(hidden)
        jax.block_until_ready(out)
    moe_full_ms = (time.time() - t0) / repeats * 1000

    # --- Profile MoE router only ---
    @jax.jit
    def run_router(hidden):
        normed = moe_layer.post_attention_layernorm(hidden)
        return moe_layer.mlp.gate(normed)

    for _ in range(warmup):
        w, i = run_router(hidden)
        jax.block_until_ready(w)

    t0 = time.time()
    for _ in range(repeats):
        w, i = run_router(hidden)
        jax.block_until_ready(w)
    router_ms = (time.time() - t0) / repeats * 1000

    # --- Profile shared expert only ---
    @jax.jit
    def run_shared(hidden):
        normed = moe_layer.post_attention_layernorm(hidden)
        return moe_layer.mlp.shared_experts(normed)

    for _ in range(warmup):
        out = run_shared(hidden)
        jax.block_until_ready(out)

    t0 = time.time()
    for _ in range(repeats):
        out = run_shared(hidden)
        jax.block_until_ready(out)
    shared_ms = (time.time() - t0) / repeats * 1000

    # --- Profile full model decode step ---
    from runner import JittedModel

    def make_kv():
        return init_kv_caches(
            batch_size=batch_size, max_seq_len=max_cache_len,
            num_layers=config.num_hidden_layers,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            dtype=dtype, mesh=mesh,
        )

    jitted = JittedModel(model)

    # Prefill + decode warmup
    prompt_len = 10
    prefill_ids = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
    prefill_pos = make_positions(prompt_len, batch_size)
    prefill_mask = make_prefill_mask(prompt_len, max_cache_len, dtype=dtype)
    prefill_cache_pos = jnp.arange(prompt_len, dtype=jnp.int32)

    decode_ids = jnp.ones((batch_size, 1), dtype=jnp.int32)
    decode_pos = jnp.array([[prompt_len]], dtype=jnp.int32)
    decode_cache_pos = jnp.array([prompt_len], dtype=jnp.int32)
    decode_mask = make_decode_mask(prompt_len, max_cache_len, dtype=dtype)

    for _ in range(warmup):
        kv_caches = make_kv()
        _, kv_caches = jitted(prefill_ids, prefill_pos, prefill_mask,
                              kv_caches=kv_caches, cache_position=prefill_cache_pos)
        logits, _ = jitted(decode_ids, decode_pos, decode_mask,
                           kv_caches=kv_caches, cache_position=decode_cache_pos)
        jax.block_until_ready(logits)

    t0 = time.time()
    for _ in range(repeats):
        kv_caches = make_kv()
        _, kv_caches = jitted(prefill_ids, prefill_pos, prefill_mask,
                              kv_caches=kv_caches, cache_position=prefill_cache_pos)
        logits, _ = jitted(decode_ids, decode_pos, decode_mask,
                           kv_caches=kv_caches, cache_position=decode_cache_pos)
        jax.block_until_ready(logits)
    full_step_ms = (time.time() - t0) / repeats * 1000

    # Measure prefill alone to subtract
    t0 = time.time()
    for _ in range(repeats):
        kv_caches = make_kv()
        _, kv_caches = jitted(prefill_ids, prefill_pos, prefill_mask,
                              kv_caches=kv_caches, cache_position=prefill_cache_pos)
        jax.block_until_ready(kv_caches[0].k)
    prefill_only_ms = (time.time() - t0) / repeats * 1000

    full_decode_ms = full_step_ms - prefill_only_ms

    # --- Derived metrics ---
    routed_ms = moe_full_ms - shared_ms - router_ms  # approximate
    other_ms = full_decode_ms - (attn_ms + moe_full_ms) * config.num_hidden_layers / config.num_hidden_layers
    # Better estimate: per-layer = attn + moe, total = N_layers * per_layer + overhead
    per_layer_ms = attn_ms + moe_full_ms
    layers_total_ms = per_layer_ms * config.num_hidden_layers
    overhead_ms = full_decode_ms - layers_total_ms

    print("\n" + "=" * 70)
    print("GLM-4.7-Flash DECODE STEP LATENCY BREAKDOWN")
    print(f"  Config: TP={mesh.shape.get('tensor',1)}, EP={mesh.shape.get('expert',1)}, batch=1")
    print("=" * 70)

    print(f"\n  PER LAYER ({config.num_hidden_layers} layers):")
    print(f"    MLA Attention:      {attn_ms:6.2f} ms")
    print(f"    MoE total:          {moe_full_ms:6.2f} ms")
    print(f"      Router (gate):    {router_ms:6.2f} ms")
    print(f"      Routed experts:   {routed_ms:6.2f} ms  (shard_map + K=4 local gather)")
    print(f"      Shared expert:    {shared_ms:6.2f} ms")
    print(f"    Layer total:        {per_layer_ms:6.2f} ms")

    print(f"\n  FULL DECODE STEP:")
    print(f"    {config.num_hidden_layers} layers × {per_layer_ms:.2f} ms = {layers_total_ms:.1f} ms")
    print(f"    Overhead (embed/norm/lmhead): {overhead_ms:.1f} ms")
    print(f"    Total:              {full_decode_ms:6.2f} ms  ({1000/full_decode_ms:.1f} tok/s)")

    print(f"\n  BREAKDOWN (% of full step):")
    print(f"    MLA Attention:      {attn_ms * config.num_hidden_layers / full_decode_ms * 100:5.1f}%")
    print(f"    MoE routed:         {routed_ms * config.num_hidden_layers / full_decode_ms * 100:5.1f}%")
    print(f"    Shared expert:      {shared_ms * config.num_hidden_layers / full_decode_ms * 100:5.1f}%")
    print(f"    Router:             {router_ms * config.num_hidden_layers / full_decode_ms * 100:5.1f}%")
    print(f"    Overhead:           {overhead_ms / full_decode_ms * 100:5.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=2)
    args = parser.parse_args()

    dtype = jnp.bfloat16
    from utils.mesh_utils import create_mesh
    mesh = create_mesh(tp=args.tp, dp=1, ep=args.ep)
    logger.info("Mesh: %s", mesh.shape)

    config = ModelConfig.from_pretrained(args.model_path)
    logger.info("Model: %s, layers=%d", config.model_type, config.num_hidden_layers)

    from models.glm4_flash import GLM4FlashForCausalLM
    model = GLM4FlashForCausalLM(config, mesh, dtype=dtype)
    model.load_weights(config)

    from utils.weight_utils import shard_model_params
    shard_model_params(model, mesh)

    profile_components(model, config, mesh, dtype)


if __name__ == "__main__":
    main()
