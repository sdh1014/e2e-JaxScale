"""Layer-0 sub-component diagnostic: JaxScale vs HuggingFace.

Uses forward hooks to capture intermediates from HF model,
and manual step-through for JaxScale.
"""

import argparse
import logging

import jax
import jax.numpy as jnp
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cos_sim(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compare(name, jax_val, hf_val):
    j = np.array(jax_val, dtype=np.float32)
    h = np.array(hf_val, dtype=np.float32)
    cos = cos_sim(j, h)
    mad = float(np.max(np.abs(j - h)))
    mean_d = float(np.mean(np.abs(j - h)))
    tag = "OK" if cos > 0.9999 else ("WARN" if cos > 0.999 else "FAIL")
    print(f"  [{tag}] {name:<40} cos={cos:.8f}  max_diff={mad:.6f}  mean_diff={mean_d:.8f}")
    return cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids_np = tokenizer.encode(input_text, return_tensors="np")
    seq_len = input_ids_np.shape[1]

    print(f"Prompt: {prompt}, Tokens: {seq_len}")

    # =====================================================================
    # HF: capture intermediates via hooks
    # =====================================================================
    print(f"\n{'='*80}\nHF: Extracting intermediates via hooks\n{'='*80}")

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to("cpu").eval()

    hf = {}

    def save(name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                hf[name] = out[0].detach().float().numpy()
            else:
                hf[name] = out.detach().float().numpy()
        return hook

    def save_input(name):
        def hook(mod, inp, out):
            hf[name] = inp[0].detach().float().numpy()
        return hook

    handles = []
    layer0 = hf_model.model.layers[0]

    # Print attention implementation type
    attn_impl = getattr(hf_model.config, '_attn_implementation', 'unknown')
    print(f"  HF attention implementation: {attn_impl}")
    print(f"  HF attention class: {type(layer0.self_attn).__name__}")

    # Embedding
    handles.append(hf_model.model.embed_tokens.register_forward_hook(save('emb')))
    # input_layernorm
    handles.append(layer0.input_layernorm.register_forward_hook(save('normed')))
    # Q/K/V projections
    handles.append(layer0.self_attn.q_proj.register_forward_hook(save('q')))
    handles.append(layer0.self_attn.k_proj.register_forward_hook(save('k')))
    handles.append(layer0.self_attn.v_proj.register_forward_hook(save('v')))
    # O projection
    handles.append(layer0.self_attn.o_proj.register_forward_hook(save('o_out')))
    handles.append(layer0.self_attn.o_proj.register_forward_hook(save_input('attn_out_flat')))
    # Full attention output
    handles.append(layer0.self_attn.register_forward_hook(save('self_attn_out')))
    # post_attention_layernorm
    handles.append(layer0.post_attention_layernorm.register_forward_hook(save('post_norm')))
    # MLP sub-components
    handles.append(layer0.mlp.gate_up_proj.register_forward_hook(save('gate_up')))
    handles.append(layer0.mlp.down_proj.register_forward_hook(save('mlp_out')))
    handles.append(layer0.mlp.down_proj.register_forward_hook(save_input('mlp_act')))
    # Full layer output
    handles.append(layer0.register_forward_hook(save('layer0_out')))
    # Layer 1-9 output
    for i in range(1, min(10, len(hf_model.model.layers))):
        handles.append(hf_model.model.layers[i].register_forward_hook(save(f'layer{i}_out')))

    input_ids_pt = torch.from_numpy(input_ids_np.astype(np.int64))
    with torch.no_grad():
        hf_output = hf_model(input_ids_pt)
        hf_logits = hf_output.logits.float().numpy()[0, -1, :]

    for h in handles:
        h.remove()

    # Split gate_up
    gate_up = hf['gate_up']
    mid = gate_up.shape[-1] // 2
    hf['gate'] = gate_up[..., :mid]
    hf['up'] = gate_up[..., mid:]

    del hf_model
    import gc; gc.collect()

    # =====================================================================
    # JaxScale: step-through layer 0
    # =====================================================================
    print(f"\n{'='*80}\nJaxScale: Extracting intermediates\n{'='*80}")

    from configs.model_config import ModelConfig
    from models.glm4 import GLM4ForCausalLM
    from runner import make_causal_mask, make_positions
    from utils.mesh_utils import create_mesh
    from utils.weight_utils import shard_model_params

    mesh = create_mesh(tp=1, dp=1, ep=1)
    config = ModelConfig.from_pretrained(args.model_path)
    model = GLM4ForCausalLM(config, mesh, dtype=jnp.bfloat16)
    model.load_weights(config)
    shard_model_params(model, mesh)

    input_ids_jax = jnp.asarray(input_ids_np)
    positions = make_positions(seq_len, 1)
    mask = make_causal_mask(seq_len, dtype=jnp.bfloat16)

    layer0 = model.model.layers[0]
    num_heads, num_kv_heads, head_dim = 32, 2, 128
    scale = head_dim ** -0.5

    # Embedding
    jax_emb = model.model.embed_tokens(input_ids_jax)

    # input_layernorm
    jax_normed = layer0.input_layernorm(jax_emb)

    # Q/K/V projections
    jax_q = layer0.self_attn.q_proj(jax_normed)
    jax_k = layer0.self_attn.k_proj(jax_normed)
    jax_v = layer0.self_attn.v_proj(jax_normed)

    # --- Manual attention step-through for granular comparison ---
    # Reshape to [B, S, H, hd]
    jax_q_r = jax_q.reshape(1, seq_len, num_heads, head_dim)
    jax_k_r = jax_k.reshape(1, seq_len, num_kv_heads, head_dim)
    jax_v_r = jax_v.reshape(1, seq_len, num_kv_heads, head_dim)

    # RoPE
    q_flat = jax_q_r.reshape(-1, num_heads, head_dim)
    k_flat = jax_k_r.reshape(-1, num_kv_heads, head_dim)
    pos_flat = positions.reshape(-1)
    q_rot, k_rot = layer0.self_attn.rotary_emb(pos_flat, q_flat, k_flat)
    jax_q_rope = q_rot.reshape(1, seq_len, num_heads, head_dim)
    jax_k_rope = k_rot.reshape(1, seq_len, num_kv_heads, head_dim)

    # GQA repeat
    jax_k_rep = jnp.repeat(jax_k_rope, num_heads // num_kv_heads, axis=2)
    jax_v_rep = jnp.repeat(jax_v_r, num_heads // num_kv_heads, axis=2)

    # Transpose to [B, H, S, hd]
    jax_q_t = jnp.transpose(jax_q_rope, (0, 2, 1, 3))
    jax_k_t = jnp.transpose(jax_k_rep, (0, 2, 1, 3))
    jax_v_t = jnp.transpose(jax_v_rep, (0, 2, 1, 3))

    # Attention scores
    jax_attn_w = jnp.matmul(jax_q_t, jnp.swapaxes(jax_k_t, -2, -1)) * scale
    jax_attn_w_masked = jax_attn_w + mask
    jax_attn_probs = jax.nn.softmax(jax_attn_w_masked.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
    jax_attn_result = jnp.matmul(jax_attn_probs, jax_v_t)

    # O projection
    jax_attn_out_flat = jnp.transpose(jax_attn_result, (0, 2, 1, 3)).reshape(1, seq_len, -1)
    jax_o_out = layer0.self_attn.o_proj(jax_attn_out_flat)

    # Full attention forward for overall comparison
    jax_attn_out, _ = layer0.self_attn(jax_normed, positions, mask)

    # After attention residual
    jax_after_attn = jax_emb + jax_attn_out

    # post_attention_layernorm
    jax_post_norm = layer0.post_attention_layernorm(jax_after_attn)

    # MLP
    jax_gate = layer0.mlp.gate_proj(jax_post_norm)
    jax_up = layer0.mlp.up_proj(jax_post_norm)
    jax_mlp_act = jax.nn.silu(jax_gate) * jax_up
    jax_mlp_out = layer0.mlp.down_proj(jax_mlp_act)

    # Layer output
    jax_layer0_out = jax_after_attn + jax_mlp_out

    # =====================================================================
    # Compare
    # =====================================================================
    print(f"\n{'='*80}\nSUB-COMPONENT COMPARISON (Layer 0, TP=1)\n{'='*80}")

    print("\n--- Embedding & Norm ---")
    compare("embedding", jax_emb, hf['emb'])
    compare("input_layernorm", jax_normed, hf['normed'])

    print("\n--- Q/K/V Projections ---")
    compare("q_proj", jax_q, hf['q'])
    compare("k_proj", jax_k, hf['k'])
    compare("v_proj", jax_v, hf['v'])

    print("\n--- Attention (full) ---")
    compare("self_attn output (via module)", jax_attn_out, hf['self_attn_out'])
    compare("o_proj input (attn_out_flat)", jax_attn_out_flat, hf['attn_out_flat'])
    compare("o_proj output (manual)", jax_o_out, hf['o_out'])

    print("\n--- RoPE Q/K (JaxScale manual vs HF-style RoPE on same Q/K) ---")
    # Compute HF-style RoPE on the HF Q/K values for reference
    hf_q_np = hf['q']  # [1, S, 4096]
    hf_k_np = hf['k']  # [1, S, 256]
    hf_q_heads = hf_q_np.reshape(1, seq_len, num_heads, head_dim)
    hf_k_heads = hf_k_np.reshape(1, seq_len, num_kv_heads, head_dim)
    # Apply our RoPE to HF Q/K values (to isolate RoPE differences)
    hf_q_jax = jnp.asarray(hf_q_heads.reshape(-1, num_heads, head_dim), dtype=jnp.bfloat16)
    hf_k_jax = jnp.asarray(hf_k_heads.reshape(-1, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    our_rope_q, our_rope_k = layer0.self_attn.rotary_emb(pos_flat, hf_q_jax, hf_k_jax)
    # Also compute HF-style RoPE (using doubled cos/sin + rotate_half)
    inv_freq_np = 1.0 / (10000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64))
    pos_np = np.arange(seq_len, dtype=np.float32)
    freqs_np = np.einsum("n,d->nd", pos_np, inv_freq_np)  # [S, 32]
    freqs_doubled = np.concatenate([freqs_np, freqs_np], axis=-1)  # [S, 64]
    cos_hf = np.cos(freqs_doubled)  # [S, 64]
    sin_hf = np.sin(freqs_doubled)  # [S, 64]
    # Apply to HF Q (first head only for comparison)
    q_head0 = hf_q_heads[0, :, 0, :]  # [S, 128]
    q_rot_part = q_head0[:, :64]  # [S, 64]
    q_pass_part = q_head0[:, 64:]  # [S, 64]
    # rotate_half
    q_rot_x1 = q_rot_part[:, :32]
    q_rot_x2 = q_rot_part[:, 32:]
    q_rotated_hf = q_rot_part * cos_hf + np.concatenate([-q_rot_x2, q_rot_x1], axis=-1) * sin_hf
    q_after_rope_hf = np.concatenate([q_rotated_hf, q_pass_part], axis=-1)  # [S, 128]
    # Compare our RoPE Q head 0 vs HF-style RoPE Q head 0
    our_rope_q_head0 = np.array(our_rope_q[:, 0, :], dtype=np.float32)  # [S, 128]
    compare("RoPE Q head0 (our impl vs HF-style)", our_rope_q_head0, q_after_rope_hf)
    # Compare our RoPE Q (on JaxScale Q) vs our RoPE Q (on HF Q)
    compare("RoPE Q (JaxScale Q input vs HF Q input)", jax_q_rope[:, :, 0, :].reshape(seq_len, head_dim), our_rope_q[:, 0, :])

    print("\n--- Post-Attention ---")
    hf_after_attn = hf['emb'] + hf['self_attn_out']
    compare("after attn residual (emb + attn)", jax_after_attn, hf_after_attn)
    compare("post_attention_layernorm", jax_post_norm, hf['post_norm'])

    print("\n--- MLP ---")
    # Compare fused vs separate gate_up
    jax_gate_up = jnp.concatenate([jax_gate, jax_up], axis=-1)
    compare("gate_up (Jax concat vs HF fused)", jax_gate_up, hf['gate_up'])
    compare("gate only", jax_gate, hf['gate'])
    compare("up only", jax_up, hf['up'])
    compare("mlp activation (silu(gate)*up)", jax_mlp_act, hf['mlp_act'])
    compare("mlp output (down_proj)", jax_mlp_out, hf['mlp_out'])

    print("\n--- Layer 0 Output ---")
    compare("layer 0 output", jax_layer0_out, hf['layer0_out'])

    # =====================================================================
    # Test: fp32 accumulation
    # =====================================================================
    print(f"\n{'='*80}\nTEST: bf16 vs fp32 accumulation (Q projection)\n{'='*80}")

    from jax import lax
    q_w = layer0.self_attn.q_proj.weight.value
    q_b = layer0.self_attn.q_proj.bias.value

    q_bf16 = lax.dot_general(
        jax_normed, q_w,
        (((jax_normed.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.bfloat16,
    ) + q_b

    q_fp32 = lax.dot_general(
        jax_normed, q_w,
        (((jax_normed.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16) + q_b

    compare("q_proj bf16 accum (current)", q_bf16, hf['q'])
    compare("q_proj fp32 accum (test)", q_fp32, hf['q'])

    # =====================================================================
    # Multi-layer trend
    # =====================================================================
    print(f"\n{'='*80}\nMULTI-LAYER DIVERGENCE (first 10 layers)\n{'='*80}")

    hidden = model.model.embed_tokens(input_ids_jax)
    for i in range(min(10, config.num_hidden_layers)):
        hidden, _ = model.model.layers[i](hidden, positions, mask)
        hf_key = f'layer{i}_out' if i > 0 else 'layer0_out'
        if hf_key in hf:
            compare(f"layer {i} output", hidden, hf[hf_key])

    # =====================================================================
    # Final logits
    # =====================================================================
    print(f"\n{'='*80}\nFINAL LOGITS\n{'='*80}")

    jax_logits, _ = model(input_ids_jax, positions, mask)
    jax_last = np.array(jax_logits[0, -1, :], dtype=np.float32)
    compare("logits (last position)", jax_last, hf_logits)

    hf_top5 = np.argsort(hf_logits)[-5:][::-1]
    jax_top5 = np.argsort(jax_last)[-5:][::-1]
    print(f"\n  {'Rank':<5} {'HF':<8} {'HF logit':<11} {'Jax':<8} {'Jax logit':<11} {'Match'}")
    for r in range(5):
        h, j = int(hf_top5[r]), int(jax_top5[r])
        print(f"  {r+1:<5} {h:<8} {hf_logits[h]:<11.3f} {j:<8} {jax_last[j]:<11.3f} {'YES' if h==j else 'NO'}")


if __name__ == "__main__":
    main()
