"""Dual optimizer setup — Muon for 2D weights + AdamW for everything else.

Partitions model parameters into two groups based on the Aero-Deuce spec:
- Group A (Muon): Internal Mamba-3 projections, Attention QKV operators,
  FFN/Expert weight matrices — all 2D weight matrices.
- Group B (AdamW): Embedding tables, all RMSNorm scales, and MoE router logits.

The router gate is explicitly excluded from Muon because orthogonalizing
router logits disrupts the load-balance signal and can cause expert collapse.
"""

import torch
from torch.optim import AdamW

from configs.base import TrainConfig
from aero_deuce.optim.muon import Muon


def create_dual_optimizer(
    model: torch.nn.Module,
    train_config: TrainConfig,
) -> tuple[list[torch.optim.Optimizer], dict[str, list[str]]]:
    """Create the dual Muon + AdamW optimizer setup.

    Args:
        model: The AeroDeuceForCausalLM model.
        train_config: Training configuration.

    Returns:
        Tuple of:
        - List of two optimizers: [Muon, AdamW]
        - Dictionary mapping group name to list of parameter names for logging
    """
    muon_params = []
    muon_names = []
    adamw_params = []
    adamw_names = []

    # Keywords that exclude a parameter from Muon (even if 2D)
    adamw_keywords = {"embed", "norm", "router", "gate"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Decision logic: 2D weight matrix that isn't in the AdamW exclusion list
        is_muon = (
            param.ndim >= 2
            and not any(kw in name for kw in adamw_keywords)
        )

        if is_muon:
            muon_params.append(param)
            muon_names.append(name)
        else:
            adamw_params.append(param)
            adamw_names.append(name)

    # Create Muon optimizer for 2D weight matrices
    optimizer_muon = Muon(
        muon_params,
        lr=train_config.learning_rate * train_config.muon_lr_scale,
        momentum=train_config.muon_momentum,
        ns_steps=train_config.muon_ns_steps,
    )

    # Create AdamW optimizer for embeddings, norms, router
    optimizer_adamw = AdamW(
        adamw_params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=train_config.adam_betas,
        eps=train_config.adam_eps,
    )

    optimizers = [optimizer_muon, optimizer_adamw]
    param_groups = {
        "muon": muon_names,
        "adamw": adamw_names,
    }

    return optimizers, param_groups
