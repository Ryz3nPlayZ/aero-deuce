"""SwiGLU Feed-Forward Network and DeepSeekMoE FFN.

Implements two FFN variants:
1. SwiGLUFFN: Dense FFN used in layers 0-2 (foundational base)
2. DeepSeekMoEFFN: Mixture of Experts FFN with shared + routed experts for layers 3-35

The DeepSeekMoE uses fine-grained expert segmentation where each routed expert
has a smaller hidden dimension, plus a shared expert that always fires.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ModelConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    forward(x) = w_down(silu(w_gate(x)) * w_up(x))

    This is the standard FFN for transformer-based LLMs, used for:
    - Dense FFN in layers 0-2
    - Individual expert networks inside DeepSeekMoE
    """

    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekMoEFFN(nn.Module):
    """DeepSeek-style Mixture of Experts FFN.

    Architecture:
    - 1 shared expert that fires on every token (always active)
    - N routed experts, with top-k selected per token by the router
    - Output = shared_expert(x) + weighted sum of top-k routed experts

    This follows the DeepSeekMoE fine-grained expert segmentation strategy,
    which achieves better performance than coarse-grained MoE by allowing
    finer specialization across experts.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.expert_hidden_dim = config.computed_expert_hidden_dim

        # Shared expert — always active for every token
        self.shared_expert = SwiGLUFFN(
            d_model=config.d_model,
            hidden_dim=self.expert_hidden_dim,
            bias=config.use_bias,
        )

        # Routed experts — top-k selected per token
        self.experts = nn.ModuleList([
            SwiGLUFFN(
                d_model=config.d_model,
                hidden_dim=self.expert_hidden_dim,
                bias=config.use_bias,
            )
            for _ in range(config.n_experts)
        ])

        # Import here to avoid circular dependency
        from aero_deuce.model.router import TopKRouter
        self.router = TopKRouter(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with expert routing.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Tuple of:
            - Output tensor of shape (B, L, D)
            - Auxiliary load-balance loss (scalar) or None
        """
        B, L, D = x.shape

        # Shared expert contribution (always active)
        shared_out = self.shared_expert(x)

        # Router: select top-k experts and compute weights + aux loss
        topk_weights, topk_indices, aux_loss = self.router(x)
        # topk_weights: (B, L, top_k), topk_indices: (B, L, top_k)

        # Dispatch tokens to selected experts and combine
        # Flatten batch+seq for easier indexing
        x_flat = x.view(-1, D)  # (B*L, D)
        topk_weights_flat = topk_weights.view(-1, self.top_k)  # (B*L, top_k)
        topk_indices_flat = topk_indices.view(-1, self.top_k)  # (B*L, top_k)

        # Accumulate weighted expert outputs
        routed_out = torch.zeros_like(x_flat)  # (B*L, D)

        for k_idx in range(self.top_k):
            # Expert indices for this top-k slot
            expert_idx = topk_indices_flat[:, k_idx]  # (B*L,)
            # Weight for this top-k slot
            weight = topk_weights_flat[:, k_idx].unsqueeze(-1)  # (B*L, 1)

            # Process each expert — gather tokens assigned to this expert
            for expert_id in range(self.n_experts):
                mask = expert_idx == expert_id  # (B*L,)
                if not mask.any():
                    continue
                # Select tokens for this expert
                expert_input = x_flat[mask]  # (n_tokens, D)
                expert_output = self.experts[expert_id](expert_input)  # (n_tokens, D)
                # Weighted accumulation
                routed_out[mask] += weight[mask] * expert_output

        routed_out = routed_out.view(B, L, D)
        return shared_out + routed_out, aux_loss
