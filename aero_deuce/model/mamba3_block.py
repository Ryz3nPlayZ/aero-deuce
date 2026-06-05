"""Mamba-3 SSM Block.

Thin wrapper around the official mamba_ssm.Mamba3 module with pre-norm RMSNorm
and residual connection. This is the primary sequence mixer for Aero-Deuce,
appearing at a 3:1 ratio to attention layers across the 36-layer stack.

Mamba-3 uses complex-valued eigenvalues over [-1,1] and a MIMO formulation
that enables state tracking, counting, and parity detection.
"""

import torch
import torch.nn as nn

from configs.base import ModelConfig
from aero_deuce.model.norms import RMSNorm


class Mamba3Block(nn.Module):
    """Mamba-3 SSM layer with pre-norm and residual connection.

    Architecture:
        x_out = x + Mamba3(RMSNorm(x))

    The Mamba-3 module handles internal projections (in_proj, out_proj),
    which are the 2D weight matrices assigned to the Muon optimizer group.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Lazy import — mamba_ssm requires CUDA, so we only import when building the model
        from mamba_ssm import Mamba3

        self.ssm = Mamba3(
            d_model=config.d_model,
            d_state=config.ssm_d_state,
            expand=config.ssm_expand,
            headdim=config.ssm_headdim,
            is_mimo=False,  # SISO mode — MIMO requires >800KB shared mem (exceeds A10G limit)
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Output tensor of shape (B, L, D) with residual connection applied.
        """
        residual = x
        x_normed = self.norm(x)
        x_out = self.ssm(x_normed)
        return residual + x_out
