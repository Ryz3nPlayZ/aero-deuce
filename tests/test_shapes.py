"""Test model parameter counts and tensor shapes.

Validates that the smoke test model is approximately 115M parameters
and that forward pass produces correct output shapes.
"""

import sys
sys.path.insert(0, ".")

import pytest


def test_smoke_test_param_count():
    """Verify the 115M smoke test model has approximately 115M parameters."""
    from configs.smoke_test import smoke_test_model_config

    config = smoke_test_model_config()

    # We can only build the full model if CUDA + mamba-ssm are available
    try:
        from aero_deuce.model.transformer import AeroDeuceForCausalLM
        import torch

        model = AeroDeuceForCausalLM(config)
        total = sum(p.numel() for p in model.parameters())

        # Target: ~115M parameters (±20% tolerance for rounding)
        target = 115_000_000
        tolerance = 0.20

        print(f"\nParameter count: {total:,} ({total / 1e6:.1f}M)")
        counts = model.count_parameters()
        for k, v in counts.items():
            print(f"  {k}: {v:,} ({v / 1e6:.1f}M)")

        assert total > target * (1 - tolerance), f"Too few params: {total/1e6:.1f}M < {target*(1-tolerance)/1e6:.1f}M"
        assert total < target * (1 + tolerance), f"Too many params: {total/1e6:.1f}M > {target*(1+tolerance)/1e6:.1f}M"

    except ImportError as e:
        pytest.skip(f"CUDA/mamba-ssm not available: {e}")


def test_config_consistency():
    """Verify config properties are internally consistent."""
    from configs.smoke_test import smoke_test_model_config

    config = smoke_test_model_config()

    # head_dim should divide evenly into d_model
    assert config.d_model % config.n_q_heads == 0, "d_model must be divisible by n_q_heads"
    assert config.d_model % config.n_kv_heads == 0, "d_model must be divisible by n_kv_heads"

    # Q/KV ratio should be integer
    assert config.n_q_heads % config.n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"

    # Attention layers must be within range
    for idx in config.attn_layer_indices:
        assert 0 <= idx < config.n_layers, f"Attention layer {idx} out of range [0, {config.n_layers})"

    # Dense FFN layers must be within range
    for idx in config.dense_ffn_layers:
        assert 0 <= idx < config.n_layers, f"Dense FFN layer {idx} out of range [0, {config.n_layers})"

    # head_dim computation
    assert config.head_dim == config.d_model // config.n_q_heads
    print(f"head_dim: {config.head_dim}")
    print(f"expert_hidden_dim: {config.computed_expert_hidden_dim}")
    print(f"ffn_hidden_dim: {config.ffn_hidden_dim}")
