"""Single Aero-Deuce transformer block.

Each block selects its mixer and FFN based on layer index:
- Mixer: Mamba3Block (33 layers) or GQAAttention (3 layers at positions 9, 21, 33)
- FFN: Dense SwiGLUFFN (layers 0-2) or DeepSeekMoEFFN (layers 3-35)

Both sub-blocks use pre-norm with RMSNorm and residual connections.
"""

import torch
import torch.nn as nn

from configs.base import ModelConfig
from aero_deuce.model.norms import RMSNorm


class AeroDeuceBlock(nn.Module):
    """Single transformer block with configurable mixer and FFN.

    Architecture:
        x = x + mixer(mixer_norm(x))
        x = x + ffn(ffn_norm(x))

    Where:
    - mixer is Mamba3Block or GQAAttention depending on layer_idx
    - ffn is SwiGLUFFN or DeepSeekMoEFFN depending on layer_idx
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Select mixer based on layer index
        if layer_idx in config.attn_layer_indices:
            from aero_deuce.model.attention import GQAAttention
            self.mixer = GQAAttention(config)
            self.is_attention = True
        else:
            from aero_deuce.model.mamba3_block import Mamba3Block
            self.mixer = Mamba3Block(config)
            self.is_attention = False

        # Select FFN based on layer index
        if layer_idx in config.dense_ffn_layers:
            from aero_deuce.model.ffn import SwiGLUFFN
            self.ffn = SwiGLUFFN(
                d_model=config.d_model,
                hidden_dim=config.ffn_hidden_dim,
                bias=config.use_bias,
            )
            self.is_moe = False
        else:
            from aero_deuce.model.ffn import DeepSeekMoEFFN
            self.ffn = DeepSeekMoEFFN(config)
            self.is_moe = True

        # Pre-norm layers
        self.mixer_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, L, D).
            freqs_cis: RoPE frequencies (only needed for attention layers).

        Returns:
            Tuple of:
            - Output tensor of shape (B, L, D)
            - MoE auxiliary loss from this layer (or None if dense FFN)
        """
        # Mixer with pre-norm and residual
        x = x + self.mixer(self.mixer_norm(x), freqs_cis=freqs_cis)

        # FFN with pre-norm and residual
        if self.is_moe:
            ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
            x = x + ffn_out
            return x, aux_loss
        else:
            x = x + self.ffn(self.ffn_norm(x))
            return x, None
