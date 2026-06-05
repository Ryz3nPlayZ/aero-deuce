"""Top-K Expert Router with load-balance auxiliary loss.

Implements the routing mechanism for DeepSeekMoE. For each token, the router
selects the top-k experts from N total routed experts. A load-balance auxiliary
loss prevents router collapse where only a few experts receive all tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ModelConfig


class TopKRouter(nn.Module):
    """Top-K expert router with load-balance auxiliary loss.

    The router uses a linear gate to produce logits for each expert,
    applies softmax, selects top-k, and renormalizes the weights.

    Load-balance loss:
        aux_loss = alpha * N * sum(f_i * P_i)
    where f_i = fraction of tokens routed to expert i,
    P_i = average router probability for expert i.

    A perfectly balanced router has aux_loss = alpha.
    An imbalanced router has aux_loss >> alpha.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_alpha = config.router_aux_loss_alpha

        # Router gate — small initialization to encourage balanced initial routing
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=config.router_init_std)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Route tokens to top-k experts.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Tuple of:
            - topk_weights: (B, L, top_k) — normalized expert weights
            - topk_indices: (B, L, top_k) — selected expert indices
            - aux_loss: scalar load-balance auxiliary loss (or None if alpha=0)
        """
        logits = self.gate(x)  # (B, L, N_experts)

        # Softmax over experts
        probs = F.softmax(logits, dim=-1)  # (B, L, N_experts)

        # Select top-k
        topk_weights, topk_indices = probs.topk(self.top_k, dim=-1)  # (B, L, top_k)

        # Renormalize selected weights so they sum to 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute auxiliary load-balance loss
        aux_loss = self._compute_aux_loss(probs, topk_indices)

        return topk_weights, topk_indices, aux_loss

    def _compute_aux_loss(
        self,
        probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute load-balance auxiliary loss.

        aux_loss = alpha * N * sum(f_i * P_i)

        where:
        - f_i = fraction of tokens routed to expert i
        - P_i = mean router probability for expert i across all tokens
        - N = number of experts
        """
        if self.aux_loss_alpha == 0:
            return None

        B, L, N = probs.shape
        n_tokens = B * L

        # f_i: fraction of tokens routed to each expert
        # Count how many times each expert appears in top-k selections
        expert_counts = torch.zeros(N, device=probs.device, dtype=probs.dtype)
        flat_indices = topk_indices.view(-1)  # (B*L*top_k,)
        ones = torch.ones_like(flat_indices, dtype=probs.dtype)
        expert_counts.scatter_add_(0, flat_indices, ones)
        f = expert_counts / (n_tokens * self.top_k)  # (N,)

        # P_i: mean router probability for each expert across all tokens
        P = probs.mean(dim=(0, 1))  # (N,)

        # Load-balance loss
        aux_loss = self.aux_loss_alpha * N * (f * P).sum()
        return aux_loss
