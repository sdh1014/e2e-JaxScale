"""Diagnose logits divergence between JaxScale and HuggingFace GLM-4-9B.

Compares intermediate outputs layer by layer to pinpoint where divergence starts.
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cosine_sim(a, b):
    a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def max_abs_diff(a, b):
    return np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))


def mean_abs_diff(a, b):
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))


def compare(name, jax_arr, hf_arr):
    j = np.array(jax_arr, dtype=np.float32)
    h = np.array(hf_arr, dtype=np.float32)
    cos = cosine_sim(j, h)
    mad = max_abs_diff(j, h)
    mean_d = mean_abs_diff(j, h)
    status = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "FAIL")
    print(f"  [{status}] {name:<35} cos={cos:.6f}  max_diff={mad:.6f}  mean_diff={mean_d:.6f}  shape={j.shape}")
    return cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids_np = tokenizer.encode(input_text, return_tensors="np")

    print(f"\nPrompt: {prompt}")
    print(f"Tokenized length: {input_ids_np.shape[1]}")
    print(f"Input IDs (first 20): {input_ids_np[0, :20].tolist()}")

    # =========================================================================
    # Part 1: HuggingFace reference (PyTorch, CPU, bf16)
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOADING HUGGINGFACE MODEL")
    print("=" * 80)

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cpu").eval()

    input_ids_pt = torch.from_numpy(input_ids_np.astype(np.int64))

    # Collect intermediate outputs via hooks
    hf_intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hf_intermediates[name] = output[0].detach().float().numpy()
            else:
                hf_intermediates[name] = output.detach().float().numpy()
        return hook

    # Register hooks on layers
    hf_model.model.embed_tokens.register_forward_hook(make_hook("embedding"))
    for i, layer in enumerate(hf_model.model.layers):
        layer.register_forward_hook(make_hook(f"layer_{i}"))
    hf_model.model.norm.register_forward_hook(make_hook("final_norm"))

    with torch.no_grad():
        hf_output = hf_model(input_ids_pt)
        hf_logits = hf_output.logits.float().numpy()  # [1, S, V]

    hf_last_logits = hf_logits[0, -1, :]  # [V]
    hf_top10 = np.argsort(hf_last_logits)[-10:][::-1]
    print(f"\nHF top-10 next tokens:")
    for idx in hf_top10:
        print(f"  {idx:>6d}  logit={hf_last_logits[idx]:>8.3f}  text={tokenizer.decode([idx])!r}")

    # Also extract specific weight values for comparison
    print("\n--- HF Weight Samples ---")
    hf_q_weight_0 = hf_model.model.layers[0].self_attn.q_proj.weight.detach().float().numpy()
    hf_q_bias_0 = hf_model.model.layers[0].self_attn.q_proj.bias.detach().float().numpy()
    hf_gate_up_0 = hf_model.model.layers[0].mlp.gate_up_proj.weight.detach().float().numpy()
    hf_norm_0 = hf_model.model.layers[0].input_layernorm.weight.detach().float().numpy()
    print(f"  q_proj.weight[0] shape: {hf_q_weight_0.shape}, mean={hf_q_weight_0.mean():.6f}")
    print(f"  q_proj.bias[0] shape: {hf_q_bias_0.shape}, mean={hf_q_bias_0.mean():.6f}")
    print(f"  gate_up_proj.weight[0] shape: {hf_gate_up_0.shape}")
    print(f"    first half mean (gate): {hf_gate_up_0[:hf_gate_up_0.shape[0]//2].mean():.6f}")
    print(f"    second half mean (up):  {hf_gate_up_0[hf_gate_up_0.shape[0]//2:].mean():.6f}")

    # Free HF model
    del hf_model
    import gc; gc.collect()

    # =========================================================================
    # Part 2: JaxScale (JAX, TPU, bf16)
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOADING JAXSCALE MODEL")
    print("=" * 80)

    from configs.model_config import ModelConfig
    from models.glm4 import GLM4ForCausalLM
    from runner import make_causal_mask, make_positions
    from utils.mesh_utils import create_mesh
    from utils.weight_utils import shard_model_params

    mesh = create_mesh(tp=args.tp, dp=1, ep=1)
    config = ModelConfig.from_pretrained(args.model_path)
    model = GLM4ForCausalLM(config, mesh, dtype=jnp.bfloat16)
    model.load_weights(config)
    shard_model_params(model, mesh)

    input_ids_jax = jnp.asarray(input_ids_np)
    seq_len = input_ids_jax.shape[1]
    positions = make_positions(seq_len, 1)
    mask = make_causal_mask(seq_len, dtype=jnp.bfloat16)

    # =========================================================================
    # Part 3: Compare weights
    # =========================================================================
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON (Layer 0)")
    print("=" * 80)

    # q_proj weight: HF is [out, in], JaxScale is [in, out]
    jax_q_w = np.array(model.model.layers[0].self_attn.q_proj.weight.value, dtype=np.float32)
    print(f"\n  JaxScale q_proj.weight shape: {jax_q_w.shape}")
    compare("q_proj.weight (Jax vs HF.T)", jax_q_w, hf_q_weight_0.T)

    jax_q_b = np.array(model.model.layers[0].self_attn.q_proj.bias.value, dtype=np.float32)
    compare("q_proj.bias", jax_q_b, hf_q_bias_0)

    # gate_proj weight: comes from first half of gate_up_proj
    jax_gate_w = np.array(model.model.layers[0].mlp.gate_proj.weight.value, dtype=np.float32)
    hf_gate_w = hf_gate_up_0[:hf_gate_up_0.shape[0]//2].T  # split first half, then transpose
    compare("gate_proj.weight (from gate_up split)", jax_gate_w, hf_gate_w)

    jax_up_w = np.array(model.model.layers[0].mlp.up_proj.weight.value, dtype=np.float32)
    hf_up_w = hf_gate_up_0[hf_gate_up_0.shape[0]//2:].T  # split second half, then transpose
    compare("up_proj.weight (from gate_up split)", jax_up_w, hf_up_w)

    jax_norm_w = np.array(model.model.layers[0].input_layernorm.scale.value, dtype=np.float32)
    compare("input_layernorm.scale", jax_norm_w, hf_norm_0)

    # =========================================================================
    # Part 4: Compare intermediate outputs layer by layer
    # =========================================================================
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER OUTPUT COMPARISON")
    print("=" * 80)

    # Run JaxScale embedding
    jax_emb = np.array(model.model.embed_tokens(input_ids_jax), dtype=np.float32)
    compare("embedding output", jax_emb, hf_intermediates["embedding"])

    # Run JaxScale layer by layer
    hidden = model.model.embed_tokens(input_ids_jax)
    first_divergent_layer = -1

    for i, layer in enumerate(model.model.layers):
        hidden, _ = layer(hidden, positions, mask)
        jax_hidden = np.array(hidden, dtype=np.float32)
        cos = compare(f"after layer {i}", jax_hidden, hf_intermediates[f"layer_{i}"])
        if cos < 0.99 and first_divergent_layer < 0:
            first_divergent_layer = i

        # Only print first 5 layers and any divergent layer
        if i >= 4 and first_divergent_layer < 0:
            print(f"  ... (skipping layers 5-{config.num_hidden_layers-1}, checking in background)")
            # Continue silently
            for j in range(i + 1, config.num_hidden_layers):
                layer_j = model.model.layers[j]
                hidden, _ = layer_j(hidden, positions, mask)
                jax_hid = np.array(hidden, dtype=np.float32)
                cos_j = cosine_sim(jax_hid, hf_intermediates[f"layer_{j}"])
                if cos_j < 0.99 and first_divergent_layer < 0:
                    first_divergent_layer = j
                    compare(f"after layer {j}", jax_hid, hf_intermediates[f"layer_{j}"])
            break

    # Final norm
    jax_final = np.array(model.model.norm(hidden), dtype=np.float32)
    compare("after final_norm", jax_final, hf_intermediates["final_norm"])

    # =========================================================================
    # Part 5: Compare final logits
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL LOGITS COMPARISON")
    print("=" * 80)

    jax_logits, _ = model(input_ids_jax, positions, mask)
    jax_logits_np = np.array(jax_logits, dtype=np.float32)
    jax_last_logits = jax_logits_np[0, -1, :]

    compare("full logits (last position)", jax_last_logits, hf_last_logits)

    # Top-10 comparison
    jax_top10 = np.argsort(jax_last_logits)[-10:][::-1]
    print(f"\n  {'Rank':<6} {'HF token':<10} {'HF logit':<12} {'Jax token':<10} {'Jax logit':<12} {'Match?'}")
    print(f"  {'-'*62}")
    for rank in range(10):
        h_idx, j_idx = hf_top10[rank], jax_top10[rank]
        h_text = tokenizer.decode([h_idx])
        j_text = tokenizer.decode([j_idx])
        match = "YES" if h_idx == j_idx else "NO"
        print(f"  {rank+1:<6} {h_idx:<10} {hf_last_logits[h_idx]:<12.3f} {j_idx:<10} {jax_last_logits[j_idx]:<12.3f} {match:<6} HF={h_text!r} Jax={j_text!r}")

    # =========================================================================
    # Part 6: Deep dive on first divergent layer
    # =========================================================================
    if first_divergent_layer >= 0:
        layer_idx = first_divergent_layer
        print(f"\n{'=' * 80}")
        print(f"DEEP DIVE: Layer {layer_idx} (first significant divergence)")
        print(f"{'=' * 80}")

        # Get input to this layer
        hidden_input = model.model.embed_tokens(input_ids_jax)
        for j in range(layer_idx):
            hidden_input, _ = model.model.layers[j](hidden_input, positions, mask)

        layer = model.model.layers[layer_idx]

        # Step through sub-components
        # 1. Input LayerNorm
        normed = layer.input_layernorm(hidden_input)
        print(f"\n  After input_layernorm:")
        print(f"    JaxScale stats: mean={np.array(normed).mean():.6f}, std={np.array(normed).std():.6f}")

        # 2. Q/K/V projections (before RoPE)
        q = layer.self_attn.q_proj(normed)
        k = layer.self_attn.k_proj(normed)
        v = layer.self_attn.v_proj(normed)
        print(f"    Q proj output: mean={np.array(q).mean():.6f}, std={np.array(q).std():.6f}, shape={q.shape}")
        print(f"    K proj output: mean={np.array(k).mean():.6f}, std={np.array(k).std():.6f}, shape={k.shape}")
        print(f"    V proj output: mean={np.array(v).mean():.6f}, std={np.array(v).std():.6f}, shape={v.shape}")

    # =========================================================================
    # Part 7: RoPE verification
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("ROPE VERIFICATION")
    print(f"{'=' * 80}")

    from layers.rotary import RotaryEmbedding
    rope = RotaryEmbedding(head_dim=128, rope_theta=10000.0, dtype=jnp.bfloat16, partial_rotary_factor=0.5)
    print(f"  rotary_dim: {rope.rotary_dim} (head_dim={rope.head_dim}, factor=0.5)")
    print(f"  inv_freq shape: {rope._inv_freq.shape}")
    print(f"  inv_freq[:5]: {rope._inv_freq[:5]}")

    # Compare with HF's RoPE inv_freq
    hf_rotary_dim = 128 // 2  # GLM-4 uses partial_rotary_factor=0.5
    hf_inv_freq = 1.0 / (10000.0 ** (np.arange(0, hf_rotary_dim, 2, dtype=np.float32) / hf_rotary_dim))
    print(f"  HF inv_freq[:5]: {hf_inv_freq[:5]}")
    print(f"  inv_freq match: {np.allclose(rope._inv_freq, hf_inv_freq)}")

    # Test RoPE with small example
    test_pos = jnp.array([0, 1, 2, 3])
    test_q = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    test_k = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
    q_rot, k_rot = rope(test_pos, test_q, test_k)
    print(f"  RoPE output Q[:, 0, :4]: {np.array(q_rot[0, 0, :4])}")
    print(f"  RoPE output Q[:, 0, 64:68] (pass-through): {np.array(q_rot[0, 0, 64:68])}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'=' * 80}")

    if first_divergent_layer >= 0:
        print(f"  First significant divergence (cos < 0.99) at: layer {first_divergent_layer}")
    else:
        print(f"  All layers have cos > 0.99 (divergence is gradual)")

    # Check if top-1 prediction matches
    hf_top1 = hf_top10[0]
    jax_top1 = jax_top10[0]
    print(f"  HF top-1: {hf_top1} ({tokenizer.decode([hf_top1])!r})")
    print(f"  Jax top-1: {jax_top1} ({tokenizer.decode([jax_top1])!r})")
    print(f"  Top-1 match: {'YES' if hf_top1 == jax_top1 else 'NO'}")

    # Count how many of top-10 overlap
    overlap = len(set(hf_top10.tolist()) & set(jax_top10.tolist()))
    print(f"  Top-10 overlap: {overlap}/10")


if __name__ == "__main__":
    main()
