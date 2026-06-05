"""Rotary Position Embeddings (RoPE).

Implements RoPE with configurable base frequency (theta). The default theta=1,000,000
enables natural context extension up to 128K tokens during downstream fine-tuning,
as specified in the Aero-Deuce architecture blueprint.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embeddings.

    Caches the complex exponentials (freqs_cis) for the full max_seq_len at init time.
    theta = 1,000,000 per the spec for long context extrapolation.
    """

    def __init__(self, dim: int, theta: float = 1_000_000.0, max_seq_len: int = 32_768):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        # Precompute and register as buffer (moves with model to device)
        freqs_cis = self._precompute_freqs_cis(max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _precompute_freqs_cis(self, seq_len: int) -> torch.Tensor:
        """Compute complex exponentials for RoPE.

        Returns tensor of shape (seq_len, dim/2) as complex64.
        """
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return freqs_cis for the first seq_len positions.

        Args:
            seq_len: Number of positions to return.

        Returns:
            Tensor of shape (seq_len, dim/2) as complex64.
        """
        return self.freqs_cis[:seq_len]


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape (..., L, n_heads, head_dim).
        freqs_cis: Complex exponentials of shape (L, head_dim/2).

    Returns:
        Tensor with RoPE applied, same shape as x.
    """
    head_dim = x.shape[-1]
    # Reshape x into pairs for complex multiplication
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape freqs_cis to broadcast: (L, 1, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(1)
    # Apply rotation via complex multiplication
    x_rotated = torch.view_as_real(x_complex * freqs_cis)
    return x_rotated.flatten(-2).type_as(x)
