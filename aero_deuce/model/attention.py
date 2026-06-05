"""Grouped-Query Attention (GQA) with QK-Norm and RoPE.

Implements the 3 attention anchor layers in Aero-Deuce at positions 9, 21, 33.
Uses a 2:1 Query-to-Key/Value head ratio (16Q:8KV for full model, 8Q:4KV for smoke test).

Key safety measures per the spec:
1. QK-Norm: Per-head RMSNorm applied to Q and K independently BEFORE attention.
2. RoPE: Rotary Position Embeddings with theta=1,000,000 for 128K context extension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ModelConfig
from aero_deuce.model.norms import RMSNorm
from aero_deuce.model.rope import RotaryEmbedding, apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads for GQA.

    Args:
        x: Tensor of shape (B, L, n_kv_heads, head_dim).
        n_rep: Number of repetitions (n_q_heads // n_kv_heads).

    Returns:
        Tensor of shape (B, L, n_kv_heads * n_rep, head_dim).
    """
    if n_rep == 1:
        return x
    B, L, n_kv_heads, head_dim = x.shape
    x = x.unsqueeze(3).expand(B, L, n_kv_heads, n_rep, head_dim)
    return x.reshape(B, L, n_kv_heads * n_rep, head_dim)


class GQAAttention(nn.Module):
    """Grouped-Query Attention with QK-Norm and RoPE.

    Architecture:
        Q, K, V = linear projections
        Q = q_norm(Q), K = k_norm(K)        # QK-Norm (per-head RMSNorm)
        Q, K = apply_rotary_emb(Q, K)        # RoPE with theta=1M
        K, V = repeat_kv(K, V)               # GQA head expansion
        out = flash_attention(Q, K, V)       # F.scaled_dot_product_attention
        out = wo(out)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_q_heads // self.n_kv_heads

        # QKV projections (no bias per spec)
        self.wq = nn.Linear(config.d_model, self.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_q_heads * self.head_dim, config.d_model, bias=False)

        # QK-Norm: per-head RMSNorm applied independently to Q and K
        # This stabilizes attention scores in deep hybrid models where SSM and
        # attention outputs may have different magnitude scales
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE with theta=1,000,000 for long context extension
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            theta=config.attn_theta,
            max_seq_len=config.max_seq_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, L, D).
            freqs_cis: Precomputed RoPE frequencies of shape (L, head_dim/2).
                       If None, will compute on the fly.

        Returns:
            Output tensor of shape (B, L, D).
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, L, self.n_q_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        # QK-Norm: per-head RMSNorm BEFORE attention
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        if freqs_cis is None:
            freqs_cis = self.rope(L)
        # freqs_cis shape: (L, head_dim/2) -> broadcast over heads
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # GQA: repeat KV heads to match Q heads
        k = repeat_kv(k, self.n_rep)  # (B, L, n_q_heads, head_dim)
        v = repeat_kv(v, self.n_rep)

        # Transpose for attention: (B, n_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with Flash Attention
        # is_causal=True enables causal masking for autoregressive LM
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)
