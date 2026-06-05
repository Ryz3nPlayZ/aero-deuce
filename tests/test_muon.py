"""Test Muon optimizer correctness.

Validates:
- Newton-Schulz orthogonalization produces approximately orthogonal matrices
- Muon optimizer step updates parameters correctly
- 2D weight matrices are processed, 1D parameters are skipped
"""

import sys
sys.path.insert(0, ".")

import pytest
import torch


def test_newton_schulz_orthogonality():
    """Verify Newton-Schulz iteration produces approximately orthogonal matrices."""
    from aero_deuce.optim.muon import newton_schulz

    torch.manual_seed(42)

    # Test with a random tall matrix (m > n)
    M_tall = torch.randn(32, 16)
    Z_tall = newton_schulz(M_tall, steps=5)

    # Z^T @ Z should be close to identity for orthogonal matrix
    ZtZ = Z_tall.float() @ Z_tall.float().mT
    identity = torch.eye(ZtZ.shape[0])
    error = (ZtZ - identity).abs().max().item()
    print(f"Tall matrix orthogonality error: {error:.6f}")
    assert error < 0.1, f"Orthogonality error too large: {error}"

    # Test with a random wide matrix (n > m)
    M_wide = torch.randn(16, 32)
    Z_wide = newton_schulz(M_wide, steps=5)
    ZtZ_wide = Z_wide.float().mT @ Z_wide.float()
    identity_wide = torch.eye(ZtZ_wide.shape[0])
    error_wide = (ZtZ_wide - identity_wide).abs().max().item()
    print(f"Wide matrix orthogonality error: {error_wide:.6f}")
    assert error_wide < 0.1, f"Orthogonality error too large: {error_wide}"

    print("✓ Newton-Schulz produces approximately orthogonal matrices")


def test_muon_optimizer_step():
    """Verify Muon optimizer updates 2D parameters and skips 1D parameters."""
    from aero_deuce.optim.muon import Muon

    torch.manual_seed(42)

    # Create a simple 2D parameter
    weight = torch.randn(16, 8, requires_grad=True)
    bias = torch.randn(16, requires_grad=True)

    # Create optimizer with both parameters
    optimizer = Muon([weight, bias], lr=0.01, momentum=0.95)

    # Simulate a gradient
    weight.grad = torch.randn_like(weight)
    bias.grad = torch.randn_like(bias)

    # Save initial values
    weight_before = weight.clone()
    bias_before = bias.clone()

    # Step
    optimizer.step()

    # Weight should have changed (2D matrix → Muon processes it)
    weight_changed = not torch.allclose(weight, weight_before)
    print(f"Weight changed after Muon step: {weight_changed}")
    assert weight_changed, "Muon should update 2D weight matrices"

    # Bias should NOT have changed (1D → Muon skips it)
    bias_unchanged = torch.allclose(bias, bias_before)
    print(f"Bias unchanged after Muon step: {bias_unchanged}")
    assert bias_unchanged, "Muon should skip 1D parameters (bias, norms)"

    print("✓ Muon correctly handles 2D vs 1D parameters")


def test_lr_schedule():
    """Test warmup + cosine decay schedule."""
    from aero_deuce.training.schedule import get_lr

    # Warmup phase
    lr_0 = get_lr(0, max_steps=1000, warmup_steps=100, lr=3e-4)
    assert lr_0 > 0, "LR should be positive during warmup"
    print(f"Step 0 LR: {lr_0:.6f}")

    # Peak LR (end of warmup)
    lr_100 = get_lr(100, max_steps=1000, warmup_steps=100, lr=3e-4)
    assert abs(lr_100 - 3e-4) < 1e-6, f"LR at end of warmup should be ~3e-4, got {lr_100}"
    print(f"Step 100 LR (peak): {lr_100:.6f}")

    # Decay phase
    lr_500 = get_lr(500, max_steps=1000, warmup_steps=100, lr=3e-4)
    assert lr_500 < lr_100, "LR should decay after warmup"
    print(f"Step 500 LR: {lr_500:.6f}")

    # End of training
    lr_999 = get_lr(999, max_steps=1000, warmup_steps=100, lr=3e-4)
    assert lr_999 < lr_500, "LR should continue decaying"
    print(f"Step 999 LR: {lr_999:.6f}")

    print("✓ LR schedule behaves correctly")
