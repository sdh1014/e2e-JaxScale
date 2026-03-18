"""Benchmark for GLM-4 inference: measures prefill/decode throughput, latency, and MFU.

Usage:
    # Single config benchmark
    python benchmark.py --model_path /path/to/glm-4-9b-chat-hf --tp 1 --dp 1

    # Sweep multiple configs (for DP/TP scaling analysis)
    python benchmark.py --model_path /path/to/glm-4-9b-chat-hf --sweep

Outputs JSON results to stdout for easy comparison across TP/DP configurations.
"""

import argparse
import json
import logging
import time

import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from models.glm4 import GLM4ForCausalLM
from runner import make_causal_mask, make_decode_mask, make_positions, make_prefill_mask
from layers.kv_cache import init_kv_caches
from utils.mesh_utils import create_mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FLOPs calculation
# ---------------------------------------------------------------------------

def estimate_flops_per_token(config: ModelConfig) -> int:
    """Estimate FLOPs for one forward pass per token (GLM-4 Dense).

    Uses the standard transformer FLOPs formula:
      Forward FLOPs ≈ 2 * P  (where P = parameter count, each param does 1 mul + 1 add)

    Broken down by component:
      - Attention QKV projection: 2 * 3 * h * h  (Q, K, V each h->h for simplicity)
      - Attention output projection: 2 * h * h
      - Attention score computation: 2 * s * h  (per token, amortized)
      - MLP: 2 * 3 * h * i  (gate_proj + up_proj + down_proj, SwiGLU has 3 linears)
      - LM Head: 2 * h * V

    Per-layer = attention_proj + attn_score + mlp
    Total = n_layers * per_layer + lm_head
    """
    h = config.hidden_size
    n_kv = config.num_key_value_heads
    n_q = config.num_attention_heads
    head_dim = config.head_dim
    i = config.intermediate_size
    V = config.vocab_size
    L = config.num_hidden_layers

    # Attention projections: Q + K + V + O
    # Q: h -> n_q * head_dim, K: h -> n_kv * head_dim, V: h -> n_kv * head_dim
    # O: n_q * head_dim -> h
    flops_attn_proj = 2 * h * (n_q * head_dim + 2 * n_kv * head_dim + n_q * head_dim)

    # MLP: gate_proj(h->i) + up_proj(h->i) + down_proj(i->h) = 3 matmuls
    flops_mlp = 2 * (h * i + h * i + i * h)  # = 6 * h * i

    flops_per_layer = flops_attn_proj + flops_mlp

    # LM Head: h -> V
    flops_lm_head = 2 * h * V

    total = L * flops_per_layer + flops_lm_head
    return total


def get_peak_flops(device_type: str) -> float | None:
    """Return theoretical peak FLOPs (bf16) for known accelerators.

    Returns None if unknown, so MFU will be skipped.
    """
    # bf16 peak TFLOPS for common accelerators
    peak_tflops = {
        "l4": 121,          # NVIDIA L4: 121 TFLOPS (bf16 with sparsity) / ~60 dense
        "a100": 312,        # NVIDIA A100 SXM: 312 TFLOPS (bf16 TF32)
        "h100": 990,        # NVIDIA H100 SXM: 990 TFLOPS (bf16)
        "tpu v5e": 197,     # TPU v5e: 197 TFLOPS (bf16)
        "tpu v6e": 410,     # TPU v6e: 410 TFLOPS (bf16) (Trillium)
    }

    device_type_lower = device_type.lower()
    for key, val in peak_tflops.items():
        if key in device_type_lower:
            return val * 1e12  # TFLOPS -> FLOPS
    return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_prefill(
    model: GLM4ForCausalLM,
    batch_size: int,
    seq_len: int,
    dtype: jnp.dtype,
    warmup: int = 2,
    repeats: int = 5,
) -> dict:
    """Benchmark prefill: full forward pass on a prompt of length seq_len.

    Returns dict with latency_ms, throughput_tok_per_sec, total_tokens.
    """
    # Synthetic input
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = make_positions(seq_len, batch_size)
    mask = make_causal_mask(seq_len, dtype=dtype)

    # Warmup (includes JIT compilation)
    for _ in range(warmup):
        logits, _ = model(input_ids, positions, mask)
        logits.block_until_ready()

    # Timed runs
    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        logits, _ = model(input_ids, positions, mask)
        logits.block_until_ready()
        latencies.append(time.perf_counter() - t0)

    total_tokens = batch_size * seq_len
    avg_latency = sum(latencies) / len(latencies)

    return {
        "phase": "prefill",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
        "avg_latency_ms": avg_latency * 1000,
        "min_latency_ms": min(latencies) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "throughput_tok_per_sec": total_tokens / avg_latency,
    }


def benchmark_decode(
    model: GLM4ForCausalLM,
    batch_size: int,
    context_len: int,
    decode_steps: int,
    dtype: jnp.dtype,
    warmup: int = 2,
    repeats: int = 5,
) -> dict:
    """Benchmark decode with KV Cache.

    Prefills context_len tokens, then measures per-step decode latency
    where each step only processes 1 new token using cached K/V.
    """
    config = model.config
    max_cache_len = context_len + decode_steps + warmup + 1

    input_ids = jnp.ones((batch_size, context_len), dtype=jnp.int32)

    def run_decode(num_steps):
        """Run prefill + num_steps decode, return per-step latencies."""
        kv_caches = init_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_cache_len,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
        )

        # Prefill
        positions = make_positions(context_len, batch_size)
        cache_pos = jnp.arange(context_len, dtype=jnp.int32)
        mask = make_prefill_mask(context_len, max_cache_len, dtype=dtype)
        logits, kv_caches = model(
            input_ids, positions, mask,
            kv_caches=kv_caches, cache_position=cache_pos,
        )
        logits.block_until_ready()

        next_token = jnp.ones((batch_size, 1), dtype=jnp.int32)
        step_latencies = []
        for s in range(num_steps):
            cur_pos = context_len + s
            positions = jnp.full((batch_size, 1), cur_pos, dtype=jnp.int32)
            cache_pos = jnp.array([cur_pos], dtype=jnp.int32)
            mask = make_decode_mask(cur_pos, max_cache_len, dtype=dtype)

            t0 = time.perf_counter()
            logits, kv_caches = model(
                next_token, positions, mask,
                kv_caches=kv_caches, cache_position=cache_pos,
            )
            logits.block_until_ready()
            step_latencies.append(time.perf_counter() - t0)

        return step_latencies

    # Warmup
    run_decode(warmup)

    # Timed runs
    step_latencies = run_decode(decode_steps)

    avg_step_latency = sum(step_latencies) / len(step_latencies)
    throughput = batch_size / avg_step_latency

    return {
        "phase": "decode",
        "batch_size": batch_size,
        "context_len": context_len,
        "decode_steps": decode_steps,
        "avg_step_latency_ms": avg_step_latency * 1000,
        "min_step_latency_ms": min(step_latencies) * 1000,
        "max_step_latency_ms": max(step_latencies) * 1000,
        "throughput_tok_per_sec": throughput,
        "step_latencies_ms": [t * 1000 for t in step_latencies],
    }


def run_benchmark(
    model: GLM4ForCausalLM,
    config: ModelConfig,
    tp: int,
    dp: int,
    dtype: jnp.dtype,
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
    decode_steps: int = 8,
    warmup: int = 2,
    repeats: int = 5,
) -> dict:
    """Run full benchmark suite and return structured results."""

    if batch_sizes is None:
        batch_sizes = [1, 2, 4]
    if seq_lens is None:
        seq_lens = [128, 256, 512]

    # Device info
    devices = jax.devices()
    device_kind = devices[0].device_kind if hasattr(devices[0], "device_kind") else str(devices[0].platform)
    num_devices = tp * dp

    # FLOPs calculation
    flops_per_token = estimate_flops_per_token(config)
    peak_flops = get_peak_flops(device_kind)

    results = {
        "metadata": {
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "tp": tp,
            "dp": dp,
            "num_devices": num_devices,
            "device_kind": device_kind,
            "dtype": str(dtype),
            "flops_per_token": flops_per_token,
            "peak_device_flops": peak_flops,
            "warmup_runs": warmup,
            "timed_runs": repeats,
        },
        "prefill": [],
        "decode": [],
    }

    # --- Prefill benchmarks ---
    for bs in batch_sizes:
        for sl in seq_lens:
            logger.info("Prefill: batch_size=%d, seq_len=%d", bs, sl)
            try:
                result = benchmark_prefill(model, bs, sl, dtype, warmup, repeats)

                # Compute MFU for prefill
                if peak_flops is not None:
                    total_flops = flops_per_token * result["total_tokens"]
                    achieved_flops = total_flops / (result["avg_latency_ms"] / 1000)
                    result["mfu"] = achieved_flops / (peak_flops * num_devices)

                results["prefill"].append(result)
                logger.info(
                    "  -> %.1f tok/s, latency=%.1f ms%s",
                    result["throughput_tok_per_sec"],
                    result["avg_latency_ms"],
                    f", MFU={result['mfu']:.2%}" if "mfu" in result else "",
                )
            except Exception as e:
                logger.warning("  -> FAILED: %s", e)
                results["prefill"].append({
                    "phase": "prefill",
                    "batch_size": bs,
                    "seq_len": sl,
                    "error": str(e),
                })

    # --- Decode benchmarks ---
    for bs in batch_sizes:
        # Use smallest seq_len as context for decode
        context_len = seq_lens[0]
        logger.info("Decode: batch_size=%d, context_len=%d, steps=%d", bs, context_len, decode_steps)
        try:
            result = benchmark_decode(model, bs, context_len, decode_steps, dtype, warmup, repeats)

            # Compute MFU for decode (per-step)
            if peak_flops is not None:
                # Each decode step processes context_len + step tokens, approximate with context_len
                avg_seq = context_len + decode_steps // 2
                per_step_flops = flops_per_token * bs  # 1 new token per batch element
                achieved_flops = per_step_flops / (result["avg_step_latency_ms"] / 1000)
                result["mfu"] = achieved_flops / (peak_flops * num_devices)

            results["decode"].append(result)
            logger.info(
                "  -> %.1f tok/s, step_latency=%.1f ms%s",
                result["throughput_tok_per_sec"],
                result["avg_step_latency_ms"],
                f", MFU={result['mfu']:.2%}" if "mfu" in result else "",
            )
        except Exception as e:
            logger.warning("  -> FAILED: %s", e)
            results["decode"].append({
                "phase": "decode",
                "batch_size": bs,
                "context_len": context_len,
                "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# Comparison / analysis utilities
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    """Print a human-readable summary table."""
    meta = results["metadata"]
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: {meta['model_type']} | TP={meta['tp']} DP={meta['dp']} "
          f"| {meta['num_devices']}x {meta['device_kind']} | {meta['dtype']}")
    print("=" * 80)

    # Prefill table
    print(f"\n{'PREFILL':^80}")
    print(f"{'Batch':>6} {'SeqLen':>7} {'Tokens':>7} {'Latency(ms)':>12} "
          f"{'Tok/s':>10} {'MFU':>8}")
    print("-" * 60)
    for r in results["prefill"]:
        if "error" in r:
            print(f"{r['batch_size']:>6} {r['seq_len']:>7} {'ERROR':>7}  {r['error']}")
            continue
        mfu_str = f"{r['mfu']:.2%}" if "mfu" in r else "N/A"
        print(f"{r['batch_size']:>6} {r['seq_len']:>7} {r['total_tokens']:>7} "
              f"{r['avg_latency_ms']:>12.1f} {r['throughput_tok_per_sec']:>10.1f} "
              f"{mfu_str:>8}")

    # Decode table
    print(f"\n{'DECODE':^80}")
    print(f"{'Batch':>6} {'CtxLen':>7} {'Steps':>6} {'Step(ms)':>10} "
          f"{'Tok/s':>10} {'MFU':>8}")
    print("-" * 60)
    for r in results["decode"]:
        if "error" in r:
            print(f"{r['batch_size']:>6} {r['context_len']:>7} {'ERROR':>6}  {r['error']}")
            continue
        mfu_str = f"{r['mfu']:.2%}" if "mfu" in r else "N/A"
        print(f"{r['batch_size']:>6} {r['context_len']:>7} {r['decode_steps']:>6} "
              f"{r['avg_step_latency_ms']:>10.1f} {r['throughput_tok_per_sec']:>10.1f} "
              f"{mfu_str:>8}")

    print("=" * 80)


def compare_results(baseline: dict, scaled: dict):
    """Compare two benchmark results and print scaling efficiency.

    Args:
        baseline: Results from the baseline run (e.g., tp=1, dp=1).
        scaled: Results from the scaled run (e.g., tp=4, dp=1).
    """
    base_meta = baseline["metadata"]
    scaled_meta = scaled["metadata"]

    print("\n" + "=" * 80)
    print("SCALING COMPARISON")
    print(f"  Baseline: TP={base_meta['tp']} DP={base_meta['dp']} ({base_meta['num_devices']} devices)")
    print(f"  Scaled:   TP={scaled_meta['tp']} DP={scaled_meta['dp']} ({scaled_meta['num_devices']} devices)")
    print("=" * 80)

    device_ratio = scaled_meta["num_devices"] / base_meta["num_devices"]

    # Compare prefill
    print(f"\n{'PREFILL SCALING':^80}")
    print(f"{'Batch':>6} {'SeqLen':>7} {'Base(tok/s)':>12} {'Scaled(tok/s)':>14} "
          f"{'Speedup':>8} {'Efficiency':>10}")
    print("-" * 65)
    for br, sr in zip(baseline["prefill"], scaled["prefill"]):
        if "error" in br or "error" in sr:
            continue
        speedup = sr["throughput_tok_per_sec"] / br["throughput_tok_per_sec"]
        efficiency = speedup / device_ratio
        print(f"{br['batch_size']:>6} {br['seq_len']:>7} "
              f"{br['throughput_tok_per_sec']:>12.1f} {sr['throughput_tok_per_sec']:>14.1f} "
              f"{speedup:>7.2f}x {efficiency:>9.1%}")

    # Compare decode
    print(f"\n{'DECODE SCALING':^80}")
    print(f"{'Batch':>6} {'CtxLen':>7} {'Base(tok/s)':>12} {'Scaled(tok/s)':>14} "
          f"{'Speedup':>8} {'Efficiency':>10}")
    print("-" * 65)
    for br, sr in zip(baseline["decode"], scaled["decode"]):
        if "error" in br or "error" in sr:
            continue
        speedup = sr["throughput_tok_per_sec"] / br["throughput_tok_per_sec"]
        efficiency = speedup / device_ratio
        print(f"{br['batch_size']:>6} {br['context_len']:>7} "
              f"{br['throughput_tok_per_sec']:>12.1f} {sr['throughput_tok_per_sec']:>14.1f} "
              f"{speedup:>7.2f}x {efficiency:>9.1%}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLM-4 Inference Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1], help="Batch sizes to benchmark")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[128, 256, 512], help="Sequence lengths")
    parser.add_argument("--decode_steps", type=int, default=8, help="Number of decode steps")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep all valid TP/DP configs for available devices",
    )
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    devices = jax.devices()
    logger.info("JAX devices: %d x %s", len(devices), devices[0])

    config = ModelConfig.from_pretrained(args.model_path)
    logger.info("Model: %s (%d layers, hidden=%d)", config.model_type, config.num_hidden_layers, config.hidden_size)
    logger.info("Estimated FLOPs/token: %.2e", estimate_flops_per_token(config))

    if args.sweep:
        # Sweep all valid TP/DP configurations
        num_devices = len(devices)
        configs_to_run = []
        for tp in [1, 2, 4, 8]:
            for dp in [1, 2, 4, 8]:
                if tp * dp <= num_devices:
                    configs_to_run.append((tp, dp))

        all_results = []
        for tp, dp in configs_to_run:
            logger.info("=" * 60)
            logger.info("Running benchmark: TP=%d, DP=%d", tp, dp)
            logger.info("=" * 60)
            mesh = create_mesh(tp=tp, dp=dp)
            model = GLM4ForCausalLM(config, mesh, dtype=dtype)
            model.load_weights(config)

            result = run_benchmark(
                model, config, tp, dp, dtype,
                batch_sizes=args.batch_sizes,
                seq_lens=args.seq_lens,
                decode_steps=args.decode_steps,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            print_summary(result)
            all_results.append(result)

        # Print scaling comparison against baseline (tp=1, dp=1)
        if len(all_results) > 1:
            baseline = all_results[0]
            for scaled in all_results[1:]:
                compare_results(baseline, scaled)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info("Results saved to %s", args.output)
    else:
        # Single configuration
        mesh = create_mesh(tp=args.tp, dp=args.dp)
        model = GLM4ForCausalLM(config, mesh, dtype=dtype)

        logger.info("Loading weights...")
        t0 = time.time()
        model.load_weights(config)
        logger.info("Weights loaded in %.2fs", time.time() - t0)

        result = run_benchmark(
            model, config, args.tp, args.dp, dtype,
            batch_sizes=args.batch_sizes,
            seq_lens=args.seq_lens,
            decode_steps=args.decode_steps,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print_summary(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
