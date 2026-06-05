"""Muon Optimizer — Matrix Orthogonalized Momentum.

Implements the Muon optimizer from KellerJordan/Muon, which applies Newton-Schulz
iteration to orthogonalize the momentum matrix before each parameter update.

Muon yields a ~2x sample efficiency boost over AdamW for 2D weight matrices,
meaning our token budget behaves like it is twice as large.

Key properties:
- Only operates on 2D weight matrices (internal projections, attention QKV, FFN weights)
- Uses Newton-Schulz iteration to find the nearest orthogonal matrix
- Nesterov-style momentum for faster convergence
- Effective LR is scaled by sqrt(max(dim_0, dim_1)) automatically
"""

import torch
from torch.optim import Optimizer


def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to approximate matrix orthogonalization.

    Finds the nearest orthogonal matrix to M using a 5th-order iteration.
    Coefficients (a=3.4445, b=-4.7750, c=2.0315) from the KellerJordan/Muon reference.

    The iteration converges for any matrix with spectral norm < sqrt(3),
    which is guaranteed by the Frobenius normalization in the caller.

    Args:
        M: Input matrix of shape (m, n).
        steps: Number of Newton-Schulz iterations (default 5).

    Returns:
        Approximately orthogonal matrix of the same shape.
    """
    # Coefficients for 5th-order Newton-Schulz iteration
    a, b, c = 3.4445, -4.7750, 2.0315

    # Normalize by Frobenius norm to ensure convergence
    X = M.bfloat16()
    X = X / (X.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.mT  # (m, m)
        B = b * A + c * (A @ A)  # (m, m)
        X = a * X + B @ X  # (m, n)

    return X


class Muon(Optimizer):
    """Muon optimizer for 2D weight matrices.

    Applies matrix orthogonalization to the momentum via Newton-Schulz iteration,
    then uses the orthogonalized update direction for the parameter update.

    The effective learning rate is automatically scaled by sqrt(max(dim_0, dim_1))
    to account for the magnitude reduction from orthogonalization.

    Args:
        params: Iterable of 2D parameter tensors.
        lr: Learning rate (will be scaled by sqrt of largest dimension).
        momentum: Momentum coefficient (default 0.95).
        nesterov: Whether to use Nesterov-style momentum (default True).
        ns_steps: Number of Newton-Schulz iterations (default 5).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        For each parameter:
        1. Update exponential moving average of gradients (momentum)
        2. Compute Nesterov momentum if enabled
        3. Apply Newton-Schulz orthogonalization
        4. Scale update by lr * sqrt(max(dim_0, dim_1))
        5. Update parameter
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Only process 2D weight matrices
                if p.dim() < 2:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]

                # Update momentum: buf = beta * buf + (1 - beta) * grad
                buf.lerp_(grad, 1 - beta)

                # Nesterov momentum: use lookahead gradient estimate
                if nesterov:
                    update = grad.lerp(buf, beta)
                else:
                    update = buf.clone()

                # Newton-Schulz orthogonalization
                update = newton_schulz(update, steps=ns_steps)

                # Scale by sqrt of largest dimension
                # This compensates for the magnitude reduction from orthogonalization
                scale = max(p.shape[0], p.shape[1]) ** 0.5

                # Update parameter
                p.add_(update, alpha=-lr * scale)

        return loss
