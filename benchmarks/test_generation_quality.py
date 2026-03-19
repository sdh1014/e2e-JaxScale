"""Test GLM-4-9B text generation quality.

Runs multiple prompts through greedy and sampling generation,
checks for repetition issues, and prints detailed analysis.
"""

import argparse
import logging
import time
import re

import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from models.glm4 import GLM4ForCausalLM
from runner import greedy_generate, sample_generate, JittedModel
from utils.mesh_utils import create_mesh
from utils.weight_utils import shard_model_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TEST_PROMPTS = [
    "Please explain what machine learning is in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Java?",
    "Describe the process of photosynthesis step by step.",
    "Tell me a story about a cat who learned to fly.",
]


def compute_repetition_metrics(text: str) -> dict:
    """Compute repetition metrics for generated text."""
    words = text.split()
    if len(words) == 0:
        return {"word_count": 0, "unique_ratio": 0, "bigram_repeat_ratio": 0,
                "trigram_repeat_ratio": 0, "longest_repeat_seq": 0}

    # Word-level unique ratio
    unique_ratio = len(set(words)) / len(words) if words else 0

    # Bigram repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bigram_repeat = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0

    # Trigram repetition
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    trigram_repeat = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0

    # Longest consecutive repeated substring (word-level)
    longest_repeat = 0
    for win_size in range(3, len(words) // 2 + 1):
        found = False
        for i in range(len(words) - 2 * win_size + 1):
            chunk = words[i:i + win_size]
            next_chunk = words[i + win_size:i + 2 * win_size]
            if chunk == next_chunk:
                longest_repeat = max(longest_repeat, win_size)
                found = True
                break
        if not found and longest_repeat > 0:
            break

    return {
        "word_count": len(words),
        "unique_ratio": unique_ratio,
        "bigram_repeat_ratio": bigram_repeat,
        "trigram_repeat_ratio": trigram_repeat,
        "longest_repeat_seq": longest_repeat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # Setup
    mesh = create_mesh(tp=args.tp, dp=1, ep=1)
    config = ModelConfig.from_pretrained(args.model_path)
    model = GLM4ForCausalLM(config, mesh, dtype=jnp.bfloat16)
    model.load_weights(config)
    shard_model_params(model, mesh)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    jitted_model = JittedModel(model)

    # Warmup JIT with a reasonable cache size
    max_cache_len = 64 + args.max_new_tokens  # approximate max prompt len + generation
    jitted_model.warmup(batch_size=1, prompt_len=64, max_cache_len=max_cache_len)

    print("\n" + "=" * 80)
    print("GLM-4-9B TEXT GENERATION QUALITY TEST")
    print("=" * 80)

    all_metrics = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─' * 80}")
        print(f"[Test {i+1}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
        print(f"{'─' * 80}")

        # Tokenize with chat template
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = jnp.asarray(tokenizer.encode(input_text, return_tensors="np"))
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        prompt_len = input_ids.shape[1]

        # Need to re-warmup if prompt is longer
        needed_cache = prompt_len + args.max_new_tokens
        if needed_cache > max_cache_len:
            max_cache_len = needed_cache
            jitted_model.warmup(batch_size=1, prompt_len=prompt_len, max_cache_len=max_cache_len)

        # Greedy generation
        t0 = time.time()
        output_ids = greedy_generate(
            jitted_model, input_ids,
            max_new_tokens=args.max_new_tokens,
            max_cache_len=needed_cache,
            eos_token_id=config.eos_token_id or tokenizer.eos_token_id,
        )
        gen_time = time.time() - t0

        generated_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)

        print(f"\n[Greedy] ({num_tokens} tokens, {gen_time:.2f}s, {num_tokens/gen_time:.1f} tok/s)")
        print(generated_text[:500])
        if len(generated_text) > 500:
            print(f"  ... ({len(generated_text)} chars total)")

        metrics = compute_repetition_metrics(generated_text)
        metrics["method"] = "greedy"
        metrics["prompt_idx"] = i
        metrics["tokens"] = num_tokens
        metrics["tok_per_sec"] = num_tokens / gen_time if gen_time > 0 else 0
        all_metrics.append(metrics)

        print(f"\n  Repetition metrics:")
        print(f"    Words: {metrics['word_count']}, Unique ratio: {metrics['unique_ratio']:.2%}")
        print(f"    Bigram repeat: {metrics['bigram_repeat_ratio']:.2%}")
        print(f"    Trigram repeat: {metrics['trigram_repeat_ratio']:.2%}")
        print(f"    Longest repeated seq: {metrics['longest_repeat_seq']} words")

        if metrics['trigram_repeat_ratio'] > 0.3:
            print("    ⚠ HIGH REPETITION DETECTED")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Prompt':<8} {'Method':<10} {'Tokens':<8} {'Tok/s':<8} {'Unique%':<10} {'Bi-rep%':<10} {'Tri-rep%':<10} {'MaxRepSeq':<10}")
    print("-" * 80)
    for m in all_metrics:
        flag = " ⚠" if m['trigram_repeat_ratio'] > 0.3 else ""
        print(f"{m['prompt_idx']+1:<8} {m['method']:<10} {m['tokens']:<8} "
              f"{m['tok_per_sec']:<8.1f} {m['unique_ratio']:<10.2%} "
              f"{m['bigram_repeat_ratio']:<10.2%} {m['trigram_repeat_ratio']:<10.2%} "
              f"{m['longest_repeat_seq']:<10}{flag}")

    has_issues = any(m['trigram_repeat_ratio'] > 0.3 for m in all_metrics)
    print(f"\nOverall: {'⚠ REPETITION ISSUES FOUND' if has_issues else 'No significant repetition detected'}")


if __name__ == "__main__":
    main()
