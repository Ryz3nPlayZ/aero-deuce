"""Count and display parameters per component.

Prints a detailed table of parameter counts for the smoke test (115M)
and full (1.7B active / 8B total) configurations.

Usage:
    python scripts/count_params.py
"""

import sys
sys.path.insert(0, ".")


def count_config_params(config, label):
    """Count parameters for a given model configuration without building the model.

    This computes parameter counts analytically to avoid needing CUDA/mamba-ssm.
    """
    D = config.d_model
    V = config.vocab_size
    N = config.n_layers
    n_experts = config.n_experts
    top_k = config.top_k
    n_q = config.n_q_heads
    n_kv = config.n_kv_heads
    head_dim = config.head_dim
    ffn_hidden = config.ffn_hidden_dim
    expert_hidden = config.computed_expert_hidden_dim

    # Embedding (tied with LM head, counted once)
    embed_params = V * D

    # Per-component counts
    total = embed_params  # + final norm + no separate LM head (tied)

    # Final RMSNorm
    total += D

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  d_model={D}, n_layers={N}, vocab={V:,}")
    print(f"  experts={n_experts}, top_k={top_k}")
    print(f"  n_q_heads={n_q}, n_kv_heads={n_kv}, head_dim={head_dim}")
    print(f"  ffn_hidden={ffn_hidden}, expert_hidden={expert_hidden}")
    print(f"{'=' * 60}")

    component_totals = {}

    # Active params = params used per forward pass (shared expert + top_k routed experts only)
    active_total = embed_params + D  # embedding + final norm

    # Count by layer type
    for i in range(N):
        layer_params = 0
        active_layer_params = 0

        # Mixer norm
        layer_params += D
        active_layer_params += D

        if i in config.attn_layer_indices:
            # Attention: wq, wk, wv, wo + q_norm, k_norm
            mixer_params = (
                D * (n_q * head_dim)  # wq
                + D * (n_kv * head_dim)  # wk
                + D * (n_kv * head_dim)  # wv
                + (n_q * head_dim) * D  # wo
                + head_dim  # q_norm
                + head_dim  # k_norm
            )
            layer_params += mixer_params
            active_layer_params += mixer_params
            mixer_type = "Attention"
        else:
            # Mamba-3: in_proj + out_proj (approximately)
            # Mamba-3 internal: d_model -> expand*d_model via in_proj, then out_proj
            d_inner = config.ssm_expand * D
            mixer_params = (
                D * (d_inner * 2)  # in_proj (projects to x, z, B, C, dt, A, etc.)
                + d_inner * D  # out_proj
            )
            # Additional SSM parameters (dt_bias, B_bias, C_bias, D skip)
            ssm_extra = d_inner + config.ssm_d_state * 2 + d_inner
            mixer_params += ssm_extra
            layer_params += mixer_params
            active_layer_params += mixer_params
            mixer_type = "Mamba-3"

        # FFN norm
        layer_params += D
        active_layer_params += D

        if i in config.dense_ffn_layers:
            # Dense SwiGLU: w_gate, w_up, w_down
            ffn_params = (
                D * ffn_hidden  # w_gate
                + D * ffn_hidden  # w_up
                + ffn_hidden * D  # w_down
            )
            layer_params += ffn_params
            active_layer_params += ffn_params
            ffn_type = "DenseFFN"
        else:
            # DeepSeekMoE: shared expert + routed experts + router
            shared_params = (
                D * expert_hidden  # w_gate
                + D * expert_hidden  # w_up
                + expert_hidden * D  # w_down
            )
            routed_params = n_experts * shared_params
            router_params = D * n_experts  # router gate
            ffn_params = shared_params + routed_params + router_params
            layer_params += ffn_params
            ffn_type = f"MoE({n_experts}exp)"

            # Active FFN params: shared + top_k routed + router
            active_ffn = shared_params + (top_k * shared_params) + router_params
            active_layer_params += active_ffn

        total += layer_params
        active_total += active_layer_params
        component_totals[f"L{i:02d} ({mixer_type:8s} + {ffn_type:15s})"] = layer_params

    component_totals["Embedding (tied)"] = embed_params
    component_totals["Final Norm"] = D

    # Sort and print
    print(f"\n{'Component':<50} {'Params':>15} {'%':>8}")
    print("-" * 75)

    for name, count in sorted(component_totals.items()):
        pct = 100.0 * count / total
        print(f"{name:<50} {count:>15,} {pct:>7.1f}%")

    print("-" * 75)
    print(f"{'TOTAL (all expert weights)':<50} {total:>15,} {'100.0%':>8}")
    print(f"{'':50} {'(' + f'{total/1e6:.1f}M' + ')':>15}")
    print(f"{'ACTIVE PER TOKEN (shared + top_k experts)':<50} {active_total:>15,}")
    print(f"{'':50} {'(' + f'{active_total/1e6:.1f}M' + ')':>15}")

    return total, active_total


if __name__ == "__main__":
    from configs.smoke_test import smoke_test_model_config
    from configs.base import ModelConfig

    # Smoke test config (115M target)
    smoke_config = smoke_test_model_config()
    count_config_params(smoke_config, "Smoke Test (115M target)")

    # Full production config
    full_config = ModelConfig()  # defaults = full model
    count_config_params(full_config, "Full Production (1.7B active / 8B total)")
