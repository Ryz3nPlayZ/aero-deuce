"""Learning rate schedule — warmup + cosine decay.

Standard warmup followed by cosine decay to a minimum learning rate.
The warmup prevents early training instability by gradually ramping up the LR,
while cosine decay provides smooth convergence toward the end of training.
"""

import math


def get_lr(step: int, max_steps: int, warmup_steps: int, lr: float, min_lr: float = 1e-5) -> float:
    """Compute learning rate for a given step using warmup + cosine decay.

    Args:
        step: Current training step (0-indexed).
        max_steps: Total number of training steps.
        warmup_steps: Number of warmup steps for linear ramp.
        lr: Peak learning rate after warmup.
        min_lr: Minimum learning rate at the end of decay.

    Returns:
        Learning rate for the current step.
    """
    # Warmup phase: linear ramp from 0 to lr
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps

    # Post-training: hold at min_lr
    if step >= max_steps:
        return min_lr

    # Cosine decay phase
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)
