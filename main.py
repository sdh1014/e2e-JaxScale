"""Entry point for GLM-4 inference.

Usage:
    python main.py --model_path /path/to/glm-4-9b-chat-hf --prompt "Hello"
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from models.glm4 import GLM4ForCausalLM
from runner import greedy_generate, JittedModel
from utils.mesh_utils import create_mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GLM-4 JAX Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism degree (MoE only)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of generation")
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decode")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0=disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (1.0=disabled)")
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    # Print device info
    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    logger.info("Number of devices: %d", len(devices))

    # Create mesh
    mesh = create_mesh(tp=args.tp, dp=args.dp, ep=args.ep)
    logger.info("Mesh shape: %s", mesh.shape)

    # Load config
    config = ModelConfig.from_pretrained(args.model_path)
    logger.info("Model: %s, layers=%d, hidden=%d, heads=%d, kv_heads=%d",
                config.model_type, config.num_hidden_layers,
                config.hidden_size, config.num_attention_heads,
                config.num_key_value_heads)

    # Initialize model (select based on model type)
    logger.info("Initializing model...")
    t0 = time.time()
    if config.is_moe:
        from models.glm4_flash import GLM4FlashForCausalLM
        model = GLM4FlashForCausalLM(config, mesh, dtype=dtype)
        logger.info("Using GLM-4.7-Flash (MoE + MLA)")
    else:
        model = GLM4ForCausalLM(config, mesh, dtype=dtype)
        logger.info("Using GLM-4-9B (Dense + GQA)")
    logger.info("Model initialized in %.2fs", time.time() - t0)

    # Load weights
    logger.info("Loading weights from %s ...", args.model_path)
    t0 = time.time()
    model.load_weights(config)
    logger.info("Weights loaded in %.2fs", time.time() - t0)

    # Shard weights across TP devices (also moves weights from CPU to device)
    from utils.weight_utils import shard_model_params
    logger.info("Sharding model params (TP=%d) ...", args.tp)
    t0 = time.time()
    shard_model_params(model, mesh)
    logger.info("Params sharded in %.2fs", time.time() - t0)

    # Benchmark mode
    if args.benchmark:
        from benchmarks.benchmark import run_benchmark, print_summary
        result = run_benchmark(
            model, config, args.tp, args.dp, dtype,
            batch_sizes=[1],
            seq_lens=[128, 256, 512],
            decode_steps=8,
        )
        print_summary(result)
        return

    # Tokenize with chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    messages = [{"role": "user", "content": args.prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    logger.info("Formatted input: %s", repr(input_text[:200]))
    input_ids = tokenizer.encode(input_text, return_tensors="np")
    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]  # [1, seq_len]
    logger.info("Input tokens: %s (len=%d)", input_ids.shape, input_ids.shape[1])

    # Generate
    logger.info("Generating (greedy, max_new_tokens=%d)...", args.max_new_tokens)

    # Debug: check first-token logits to verify weight correctness
    from runner import make_causal_mask, make_positions
    seq_len = input_ids.shape[1]
    positions = make_positions(seq_len, input_ids.shape[0])
    mask = make_causal_mask(seq_len, dtype=dtype)
    logits, _ = model(input_ids, positions, mask)
    first_logits = logits[0, -1, :]  # last position logits
    top5_indices = jnp.argsort(first_logits)[-5:][::-1]
    top5_values = first_logits[top5_indices]
    logger.info("Top-5 next tokens:")
    for idx, val in zip(top5_indices.tolist(), top5_values.tolist()):
        token_str = tokenizer.decode([idx])
        logger.info("  token_id=%d, logit=%.4f, text=%r", idx, val, token_str)

    # JIT compile model for inference
    jitted_model = JittedModel(model)
    max_cache_len = seq_len + args.max_new_tokens
    jitted_model.warmup(batch_size=1, prompt_len=seq_len, max_cache_len=max_cache_len)

    # Use all EOS token IDs from generation_config (GLM-4 has 3)
    eos_token_ids = getattr(model.config, 'eos_token_id', None) or tokenizer.eos_token_id
    logger.info("EOS token IDs: %s", eos_token_ids)

    t0 = time.time()
    if args.sample:
        from runner import sample_generate
        logger.info("Generating (sample, T=%.1f, top_p=%.1f, top_k=%d, rep_pen=%.1f)...",
                     args.temperature, args.top_p, args.top_k, args.repetition_penalty)
        output_ids = sample_generate(
            jitted_model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            max_cache_len=max_cache_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos_token_ids,
        )
    else:
        output_ids = greedy_generate(
            jitted_model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            max_cache_len=max_cache_len,
            eos_token_id=eos_token_ids,
        )
    gen_time = time.time() - t0

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
    logger.info("Generated %d tokens in %.2fs (%.1f tok/s)",
                num_new_tokens, gen_time,
                num_new_tokens / gen_time if gen_time > 0 else 0)

    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Output: {generated_text}")
    print("=" * 60)


if __name__ == "__main__":
    main()
