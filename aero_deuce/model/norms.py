"""Root Mean Square Layer Normalization (RMSNorm).

Used as the pre-norm before every mixer and FFN block, and for per-head QK-Norm
inside the attention layers. No bias — matches the Aero-Deuce specification.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm: normalize by root mean square, then scale.

    Equivalent to Llama-style RMSNorm. No bias term.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
