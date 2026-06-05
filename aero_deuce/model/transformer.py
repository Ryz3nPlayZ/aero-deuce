"""Aero-Deuce For Causal Language Modeling.

Full model assembly: embedding → N × AeroDeuceBlock → RMSNorm → LM head.
Input embeddings are tied with the output LM head, saving ~311M parameters
in the full model configuration (vocab_size × d_model).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ModelConfig
from aero_deuce.model.norms import RMSNorm
from aero_deuce.model.block import AeroDeuceBlock


class AeroDeuceForCausalLM(nn.Module):
    """Aero-Deuce causal language model.

    Architecture (36 layers for full model, 16 for smoke test):
        embed_tokens(input_ids)
        for each layer:
            x = x + mixer(mixer_norm(x))    # Mamba-3 or GQA Attention
            x = x + ffn(ffn_norm(x))         # Dense SwiGLU or DeepSeekMoE
        RMSNorm(x)
        lm_head(x) -> logits

    Key features:
    - Tied embeddings (input embed = output LM head)
    - No bias anywhere (bias=False on all linear layers)
    - QK-Norm in attention layers
    - RoPE with theta=1M for long context
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            AeroDeuceBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # LM head — tied with embedding weights
        if config.tie_embeddings:
            self.lm_head = None  # Will use embed_tokens directly
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def get_lm_head(self) -> nn.Module:
        """Get the LM head module (tied embedding or separate linear)."""
        if self.lm_head is None:
            return self.embed_tokens
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (B, L).
            labels: Target token IDs of shape (B, L) for loss computation.

        Returns:
            Dictionary with:
            - "logits": (B, L, vocab_size)
            - "loss": scalar cross-entropy loss (or None if no labels)
            - "aux_loss": sum of MoE load-balance auxiliary losses
        """
        B, L = input_ids.shape

        # Embedding lookup
        x = self.embed_tokens(input_ids)  # (B, L, D)

        # Precompute RoPE for attention layers
        freqs_cis = None
        # Only compute if we have attention layers (optimization for pure SSM blocks)
        if self.config.attn_layer_indices:
            from aero_deuce.model.rope import RotaryEmbedding
            # Use the first attention layer's RoPE (they all share the same config)
            attn_layer = self.layers[self.config.attn_layer_indices[0]]
            if hasattr(attn_layer, "mixer") and hasattr(attn_layer.mixer, "rope"):
                freqs_cis = attn_layer.mixer.rope(L)

        # Forward through all layers
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, aux_loss = layer(x, freqs_cis=freqs_cis)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        # Final norm
        x = self.norm(x)

        # LM head (tied with embeddings)
        logits = self.get_lm_head()(x)  # (B, L, vocab_size)

        # Compute loss
        loss = None
        if labels is not None:
            # Shift for next-token prediction: logits[:-1] vs labels[1:]
            # But with packed sequences, labels are already the shifted targets
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            # Include MoE auxiliary loss
            loss = loss + total_aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss if labels is not None else None,
        }

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following standard transformer practices."""
        if isinstance(module, nn.Linear):
            # Standard normal initialization with std = 1/sqrt(d_model)
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component for validation.

        Returns a dictionary mapping component name to parameter count.
        """
        counts = {}
        counts["embed_tokens"] = sum(p.numel() for p in self.embed_tokens.parameters())
        counts["norm"] = sum(p.numel() for p in self.norm.parameters())
        counts["lm_head"] = 0 if self.lm_head is None else counts["embed_tokens"]

        for i, layer in enumerate(self.layers):
            layer_count = sum(p.numel() for p in layer.parameters())
            counts[f"layer_{i}"] = layer_count

        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
