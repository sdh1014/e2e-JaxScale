"""Verify KV Cache correctness and performance.

Compares greedy_generate (with KV Cache) vs greedy_generate_no_cache
to ensure identical outputs, then measures decode speedup.

Usage:
    python verify_kvcache.py --model_path /path/to/glm-4-9b-chat-hf
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from models.glm4 import GLM4ForCausalLM
from runner import (
    JittedModel,
    greedy_generate,
    greedy_generate_no_cache,
    make_causal_mask,
    make_positions,
    make_prefill_mask,
)
from layers.kv_cache import init_kv_caches
from utils.mesh_utils import create_mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def verify_prefill_equivalence(model, config, prompt_len=32, dtype=jnp.bfloat16):
    """Verify that prefill with and without cache produce identical logits."""
    logger.info("--- Prefill equivalence test (prompt_len=%d) ---", prompt_len)

    input_ids = jnp.ones((1, prompt_len), dtype=jnp.int32)
    positions = make_positions(prompt_len, 1)

    # Without cache
    mask_no_cache = make_causal_mask(prompt_len, dtype=dtype)
    logits_no_cache, _ = model(input_ids, positions, mask_no_cache)

    # With cache
    max_cache_len = prompt_len + 64
    kv_caches = init_kv_caches(
        batch_size=1,
        max_seq_len=max_cache_len,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
    )
    cache_position = jnp.arange(prompt_len, dtype=jnp.int32)
    mask_cached = make_prefill_mask(prompt_len, max_cache_len, dtype=dtype)
    logits_cached, _ = model(
        input_ids, positions, mask_cached,
        kv_caches=kv_caches, cache_position=cache_position,
    )

    # Compare
    max_diff = jnp.max(jnp.abs(logits_no_cache - logits_cached)).item()
    last_tok_diff = jnp.max(jnp.abs(logits_no_cache[0, -1] - logits_cached[0, -1])).item()
    match_argmax = jnp.array_equal(
        jnp.argmax(logits_no_cache[0, -1]), jnp.argmax(logits_cached[0, -1])
    )

    logger.info("  Max logit diff (all positions): %.6e", max_diff)
    logger.info("  Max logit diff (last position): %.6e", last_tok_diff)
    logger.info("  Argmax match (last position):   %s", match_argmax)

    passed = max_diff < 1e-3
    logger.info("  Result: %s", "PASS" if passed else "FAIL")
    return passed


def verify_generation_equivalence(model, max_new_tokens=16):
    """Verify that cached and non-cached generation produce identical token sequences."""
    logger.info("--- Generation equivalence test (max_new_tokens=%d) ---", max_new_tokens)

    input_ids = jnp.ones((1, 8), dtype=jnp.int32)

    output_cached = greedy_generate(model, input_ids, max_new_tokens=max_new_tokens)
    output_no_cache = greedy_generate_no_cache(model, input_ids, max_new_tokens=max_new_tokens)

    match = jnp.array_equal(output_cached, output_no_cache)
    logger.info("  Cached output shape:    %s", output_cached.shape)
    logger.info("  No-cache output shape:  %s", output_no_cache.shape)
    logger.info("  Token-level match:      %s", match)

    if not match:
        # Find first divergence
        min_len = min(output_cached.shape[1], output_no_cache.shape[1])
        for i in range(min_len):
            if output_cached[0, i] != output_no_cache[0, i]:
                logger.info("  First mismatch at position %d: cached=%d, no_cache=%d",
                            i, output_cached[0, i].item(), output_no_cache[0, i].item())
                break

    logger.info("  Result: %s", "PASS" if match else "FAIL")
    return match


def benchmark_decode_comparison(model, config, prompt_len=128, decode_steps=32, dtype=jnp.bfloat16):
    """Benchmark decode speed: JIT-compiled cached vs non-cached."""
    logger.info("--- Decode performance comparison (prompt=%d, steps=%d) ---", prompt_len, decode_steps)

    input_ids = jnp.ones((1, prompt_len), dtype=jnp.int32)
    max_cache_len = prompt_len + decode_steps

    # ---- JIT + KV Cache path ----
    logger.info("  Creating JittedModel and precompiling...")
    jitted_model = JittedModel(model)
    jitted_model.warmup(batch_size=1, prompt_len=prompt_len, max_cache_len=max_cache_len)

    # Extra warmup run to stabilize
    out = greedy_generate(jitted_model, input_ids, max_new_tokens=decode_steps,
                          max_cache_len=max_cache_len)
    jax.block_until_ready(out)

    # Timed runs (best of 3)
    cached_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = greedy_generate(jitted_model, input_ids, max_new_tokens=decode_steps,
                              max_cache_len=max_cache_len)
        jax.block_until_ready(out)
        cached_times.append(time.perf_counter() - t0)
    cached_time = min(cached_times)

    # ---- No-cache path (for comparison) ----
    logger.info("  Warming up no-cache path...")
    for _ in range(2):
        out = greedy_generate_no_cache(model, input_ids, max_new_tokens=4)
        jax.block_until_ready(out)

    t0 = time.perf_counter()
    out = greedy_generate_no_cache(model, input_ids, max_new_tokens=decode_steps)
    jax.block_until_ready(out)
    no_cache_time = time.perf_counter() - t0

    speedup = no_cache_time / cached_time if cached_time > 0 else 0
    cached_tok_per_sec = decode_steps / cached_time

    logger.info("  JIT + KV Cache:   %.2fs (%.1f tok/s) [best of 3, times: %s]",
                cached_time, cached_tok_per_sec,
                ", ".join(f"{t:.2f}s" for t in cached_times))
    logger.info("  Without KV Cache: %.2fs (%.1f tok/s)", no_cache_time, decode_steps / no_cache_time)
    logger.info("  Speedup:          %.2fx", speedup)

    return speedup, cached_tok_per_sec


def main():
    parser = argparse.ArgumentParser(description="Verify KV Cache correctness and performance")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    devices = jax.devices()
    logger.info("JAX devices: %d x %s", len(devices), devices[0])

    mesh = create_mesh(tp=1, dp=1)
    config = ModelConfig.from_pretrained(args.model_path)
    logger.info("Model: %s (%d layers, hidden=%d, kv_heads=%d)",
                config.model_type, config.num_hidden_layers, config.hidden_size,
                config.num_key_value_heads)

    logger.info("Initializing model...")
    model = GLM4ForCausalLM(config, mesh, dtype=dtype)
    logger.info("Loading weights...")
    t0 = time.time()
    model.load_weights(config)
    logger.info("Weights loaded in %.2fs", time.time() - t0)

    print("\n" + "=" * 60)
    print("KV CACHE VERIFICATION")
    print("=" * 60)

    # Test 1: Prefill equivalence
    test1 = verify_prefill_equivalence(model, config, prompt_len=32, dtype=dtype)

    # Test 2: Generation equivalence
    test2 = verify_generation_equivalence(model, max_new_tokens=16)

    # Test 3: Performance comparison (with proper JIT warmup)
    speedup, tok_per_sec = benchmark_decode_comparison(model, config, prompt_len=128, decode_steps=32, dtype=dtype)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Prefill equivalence:    {'PASS' if test1 else 'FAIL'}")
    print(f"  Generation equivalence: {'PASS' if test2 else 'FAIL'}")
    print(f"  KV Cache throughput:    {tok_per_sec:.1f} tok/s (best of 3, JIT excluded)")
    print(f"  Decode speedup:         {speedup:.2f}x")
    print("=" * 60)

    if not (test1 and test2):
        exit(1)


if __name__ == "__main__":
    main()
