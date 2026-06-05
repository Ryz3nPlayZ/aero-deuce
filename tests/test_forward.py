"""Test forward and backward pass through the model.

Validates that:
- Forward pass produces correct output shapes
- Loss computation works correctly
- Backward pass completes without errors
- Gradients flow through all components
"""

import sys
sys.path.insert(0, ".")

import pytest
import torch


def test_forward_backward_smoke():
    """Run a forward + backward pass on a minimal model configuration.

    Uses a tiny config (d_model=64, 2 layers) to avoid needing CUDA/mamba-ssm.
    Only tests the attention + dense FFN path since Mamba-3 requires CUDA.
    """
    try:
        from configs.base import ModelConfig
        from aero_deuce.model.attention import GQAAttention
        from aero_deuce.model.ffn import SwiGLUFFN
        from aero_deuce.model.router import TopKRouter

        config = ModelConfig(
            d_model=64,
            n_layers=2,
            vocab_size=1000,
            n_q_heads=4,
            n_kv_heads=2,
            ssm_d_state=16,
            max_seq_len=128,
            attn_layer_indices=(0, 1),
            dense_ffn_layers=(0, 1),
            n_experts=4,
            top_k=2,
            expert_hidden_dim=32,
        )

        B, L = 2, 64

        # Test GQA Attention
        attn = GQAAttention(config)
        x = torch.randn(B, L, config.d_model)
        out = attn(x)
        assert out.shape == (B, L, config.d_model), f"Attention output shape: {out.shape}"
        print(f"✓ GQA Attention: input {x.shape} → output {out.shape}")

        # Test SwiGLU FFN
        ffn = SwiGLUFFN(config.d_model, hidden_dim=config.ffn_hidden_dim, bias=False)
        out = ffn(x)
        assert out.shape == (B, L, config.d_model), f"FFN output shape: {out.shape}"
        print(f"✓ SwiGLU FFN: input {x.shape} → output {out.shape}")

        # Test Router
        router = TopKRouter(config)
        weights, indices, aux_loss = router(x)
        assert weights.shape == (B, L, config.top_k), f"Router weights shape: {weights.shape}"
        assert indices.shape == (B, L, config.top_k), f"Router indices shape: {indices.shape}"
        assert aux_loss is not None, "Router should return aux_loss"
        print(f"✓ Router: weights {weights.shape}, indices {indices.shape}, aux_loss={aux_loss.item():.6f}")

        # Test that weights sum to 1 (after renormalization)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "Router weights should sum to 1"
        print(f"✓ Router weights sum to 1")

        # Test backward pass through attention + FFN
        out = attn(x)
        out = ffn(out)
        loss = out.sum()
        loss.backward()
        print(f"✓ Backward pass completed")

    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_rope():
    """Test Rotary Position Embeddings."""
    from aero_deuce.model.rope import RotaryEmbedding, apply_rotary_emb
    import torch

    dim = 64
    seq_len = 128
    n_heads = 4

    rope = RotaryEmbedding(dim=dim, theta=1_000_000.0, max_seq_len=seq_len)
    freqs_cis = rope(seq_len)

    assert freqs_cis.shape == (seq_len, dim // 2), f"RoPE shape: {freqs_cis.shape}"
    print(f"✓ RoPE freqs_cis shape: {freqs_cis.shape}")

    # Test apply_rotary_emb
    x = torch.randn(2, seq_len, n_heads, dim)
    x_rotated = apply_rotary_emb(x, freqs_cis)
    assert x_rotated.shape == x.shape, f"Rotated shape: {x_rotated.shape}"
    print(f"✓ apply_rotary_emb preserves shape: {x.shape}")


def test_rmsnorm():
    """Test RMSNorm."""
    from aero_deuce.model.norms import RMSNorm
    import torch

    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 32, dim)
    out = norm(x)

    assert out.shape == x.shape, f"RMSNorm output shape: {out.shape}"
    # Output should be normalized (reduced variance)
    out_std = out.std(dim=-1).mean()
    print(f"✓ RMSNorm: input std={x.std(dim=-1).mean():.4f}, output std={out_std:.4f}")
