"""Compare JaxScale vs HuggingFace GLM-4-9B generation outputs.

Runs the same prompts through both implementations with greedy decoding
to check if repetition issues are from our code or the model itself.
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import torch

from configs.model_config import ModelConfig
from models.glm4 import GLM4ForCausalLM
from runner import greedy_generate, JittedModel
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
    words = text.split()
    if len(words) == 0:
        return {"word_count": 0, "unique_ratio": 0, "bigram_repeat_ratio": 0,
                "trigram_repeat_ratio": 0, "longest_repeat_seq": 0}
    unique_ratio = len(set(words)) / len(words)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bigram_repeat = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    trigram_repeat = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0
    longest_repeat = 0
    for win_size in range(3, len(words) // 2 + 1):
        found = False
        for i in range(len(words) - 2 * win_size + 1):
            if words[i:i + win_size] == words[i + win_size:i + 2 * win_size]:
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


def run_hf(model_path, tokenizer, prompts, max_new_tokens):
    """Run HuggingFace PyTorch reference implementation."""
    from transformers import AutoModelForCausalLM

    logger.info("Loading HuggingFace model (torch, bf16, CPU)...")
    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cpu")
    hf_model.eval()
    logger.info("HF model loaded in %.1fs", time.time() - t0)

    results = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            output = hf_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                temperature=1.0,
            )
        gen_time = time.time() - t0

        generated_ids = output[0, prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
        results.append({
            "prompt_idx": i,
            "text": text,
            "tokens": num_tokens,
            "tok_per_sec": num_tokens / gen_time if gen_time > 0 else 0,
            "gen_time": gen_time,
        })
        logger.info("[HF %d/%d] %d tokens in %.1fs", i + 1, len(prompts), num_tokens, gen_time)

    # Free HF model memory
    del hf_model
    import gc; gc.collect()
    return results


def run_jaxscale(model_path, tokenizer, prompts, max_new_tokens, tp):
    """Run JaxScale implementation."""
    mesh = create_mesh(tp=tp, dp=1, ep=1)
    config = ModelConfig.from_pretrained(model_path)
    model = GLM4ForCausalLM(config, mesh, dtype=jnp.bfloat16)
    model.load_weights(config)
    shard_model_params(model, mesh)

    jitted_model = JittedModel(model)
    max_cache_len = 64 + max_new_tokens
    jitted_model.warmup(batch_size=1, prompt_len=64, max_cache_len=max_cache_len)

    eos_ids = config.eos_token_id or tokenizer.eos_token_id
    logger.info("JaxScale EOS token IDs: %s", eos_ids)

    results = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = jnp.asarray(tokenizer.encode(input_text, return_tensors="np"))
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        prompt_len = input_ids.shape[1]

        needed_cache = prompt_len + max_new_tokens
        if needed_cache > max_cache_len:
            max_cache_len = needed_cache
            jitted_model.warmup(batch_size=1, prompt_len=prompt_len, max_cache_len=max_cache_len)

        t0 = time.time()
        output_ids = greedy_generate(
            jitted_model, input_ids,
            max_new_tokens=max_new_tokens,
            max_cache_len=needed_cache,
            eos_token_id=eos_ids,
        )
        gen_time = time.time() - t0

        generated_ids = output_ids[0, prompt_len:]
        text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        num_tokens = int(output_ids.shape[1] - prompt_len)
        results.append({
            "prompt_idx": i,
            "text": text,
            "tokens": num_tokens,
            "tok_per_sec": num_tokens / gen_time if gen_time > 0 else 0,
            "gen_time": gen_time,
        })
        logger.info("[JaxScale %d/%d] %d tokens in %.1fs", i + 1, len(prompts), num_tokens, gen_time)

    return results


def compare_tokens(hf_text, jax_text):
    """Compare two texts token by token and find first divergence."""
    hf_words = hf_text.split()
    jax_words = jax_text.split()
    match_count = 0
    for h, j in zip(hf_words, jax_words):
        if h == j:
            match_count += 1
        else:
            break
    total = max(len(hf_words), len(jax_words))
    return match_count, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_prompts", type=int, default=len(TEST_PROMPTS))
    args = parser.parse_args()

    prompts = TEST_PROMPTS[:args.num_prompts]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Run HuggingFace first (before JaxScale takes device memory)
    hf_results = run_hf(args.model_path, tokenizer, prompts, args.max_new_tokens)

    # Run JaxScale
    jax_results = run_jaxscale(args.model_path, tokenizer, prompts, args.max_new_tokens, args.tp)

    # Compare
    print("\n" + "=" * 100)
    print("JAXSCALE vs HUGGINGFACE COMPARISON (greedy decode)")
    print("=" * 100)

    for i, prompt in enumerate(prompts):
        hf = hf_results[i]
        jax_r = jax_results[i]
        hf_m = compute_repetition_metrics(hf["text"])
        jax_m = compute_repetition_metrics(jax_r["text"])
        match_words, total_words = compare_tokens(hf["text"], jax_r["text"])

        print(f"\n{'─' * 100}")
        print(f"[Prompt {i+1}] {prompt}")
        print(f"{'─' * 100}")

        print(f"\n  HuggingFace ({hf['tokens']} tok, {hf['tok_per_sec']:.1f} tok/s):")
        print(f"    {hf['text']}")

        print(f"\n  JaxScale ({jax_r['tokens']} tok, {jax_r['tok_per_sec']:.1f} tok/s):")
        print(f"    {jax_r['text']}")

        print(f"\n  Word-level match: {match_words}/{total_words} words match before first divergence")
        exact = "YES" if hf["text"].strip() == jax_r["text"].strip() else "NO"
        print(f"  Exact match: {exact}")

        print(f"\n  {'Metric':<25} {'HuggingFace':<15} {'JaxScale':<15} {'Diff':<10}")
        print(f"  {'-'*65}")
        print(f"  {'Trigram repeat %':<25} {hf_m['trigram_repeat_ratio']:<15.2%} {jax_m['trigram_repeat_ratio']:<15.2%} {jax_m['trigram_repeat_ratio'] - hf_m['trigram_repeat_ratio']:+.2%}")
        print(f"  {'Bigram repeat %':<25} {hf_m['bigram_repeat_ratio']:<15.2%} {jax_m['bigram_repeat_ratio']:<15.2%} {jax_m['bigram_repeat_ratio'] - hf_m['bigram_repeat_ratio']:+.2%}")
        print(f"  {'Unique word ratio':<25} {hf_m['unique_ratio']:<15.2%} {jax_m['unique_ratio']:<15.2%} {jax_m['unique_ratio'] - hf_m['unique_ratio']:+.2%}")
        print(f"  {'Longest repeat seq':<25} {hf_m['longest_repeat_seq']:<15} {jax_m['longest_repeat_seq']:<15}")

    # Summary table
    print(f"\n{'=' * 100}")
    print("SUMMARY TABLE")
    print(f"{'=' * 100}")
    print(f"{'Prompt':<8} {'HF Tri-rep%':<14} {'Jax Tri-rep%':<14} {'Diff':<10} {'Word Match':<15} {'Exact?':<8}")
    print("-" * 70)
    for i in range(len(prompts)):
        hf_m = compute_repetition_metrics(hf_results[i]["text"])
        jax_m = compute_repetition_metrics(jax_results[i]["text"])
        match_w, total_w = compare_tokens(hf_results[i]["text"], jax_results[i]["text"])
        exact = "YES" if hf_results[i]["text"].strip() == jax_results[i]["text"].strip() else "NO"
        diff = jax_m["trigram_repeat_ratio"] - hf_m["trigram_repeat_ratio"]
        print(f"{i+1:<8} {hf_m['trigram_repeat_ratio']:<14.2%} {jax_m['trigram_repeat_ratio']:<14.2%} {diff:<+10.2%} {match_w}/{total_w:<12} {exact:<8}")

    # Verdict
    hf_avg = sum(compute_repetition_metrics(r["text"])["trigram_repeat_ratio"] for r in hf_results) / len(hf_results)
    jax_avg = sum(compute_repetition_metrics(r["text"])["trigram_repeat_ratio"] for r in jax_results) / len(jax_results)
    print(f"\nAverage trigram repeat: HF={hf_avg:.2%}, JaxScale={jax_avg:.2%}")
    if abs(jax_avg - hf_avg) < 0.05:
        print("Verdict: Repetition levels are SIMILAR - issue is from the model, not JaxScale.")
    elif jax_avg > hf_avg:
        print("Verdict: JaxScale has HIGHER repetition - likely a bug in our implementation.")
    else:
        print("Verdict: JaxScale has LOWER repetition than HF reference.")


if __name__ == "__main__":
    main()
