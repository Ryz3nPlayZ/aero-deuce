"""Dual optimizer setup — Muon for LoRA matrices + AdamW for remaining params.

Partitions trainable LoRA parameters into two groups:
- Group A (Muon): LoRA A and B weight matrices (lora_A.weight, lora_B.weight).
  These are 2D matrices — exactly what Muon's Newton-Schulz orthogonalization
  was designed for. Accelerates convergence on low-rank adapter weights.
- Group B (AdamW): Non-2D trainable params (LoRA scaling vectors, embedding
  adapters if present, any 1D parameters).

Base model weights are frozen (4-bit quantized) and never appear in either group.
"""

import torch
from torch.optim import AdamW

from configs.base import TrainConfig
from aero_deuce.optim.muon import Muon


def create_dual_optimizer(
    model: torch.nn.Module,
    train_config: TrainConfig,
) -> tuple[list[torch.optim.Optimizer], dict[str, list[str]]]:
    """Create the dual Muon + AdamW optimizer setup for LoRA parameters.

    Args:
        model: The PEFT-wrapped model with LoRA adapters injected.
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
    # - norm: RMSNorm/LayerNorm scales (1D or 2D but shouldn't be orthogonalized)
    # - embed: embedding tables (if somehow trainable)
    # - lora_embedding: LoRA embedding adapters (different structure than A/B matrices)
    adamw_keywords = {"norm", "embed", "lora_embedding"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Muon targets: 2D LoRA weight matrices (lora_A.weight, lora_B.weight)
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

    print(f"[DualOptim] Muon params: {len(muon_params)} tensors")
    print(f"[DualOptim] AdamW params: {len(adamw_params)} tensors")

    if muon_params:
        total_muon = sum(p.numel() for p in muon_params)
        print(f"[DualOptim] Muon total: {total_muon:,} parameters")
    if adamw_params:
        total_adamw = sum(p.numel() for p in adamw_params)
        print(f"[DualOptim] AdamW total: {total_adamw:,} parameters")

    optimizers = []

    # Create Muon optimizer for 2D LoRA weight matrices
    if muon_params:
        optimizer_muon = Muon(
            muon_params,
            lr=train_config.learning_rate * train_config.muon_lr_scale,
            momentum=train_config.muon_momentum,
            ns_steps=train_config.muon_ns_steps,
        )
        optimizers.append(optimizer_muon)

    # Create AdamW optimizer for non-2D trainable params
    # (may be empty if all trainable params are 2D LoRA matrices)
    if adamw_params:
        optimizer_adamw = AdamW(
            adamw_params,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=train_config.adam_betas,
            eps=train_config.adam_eps,
        )
        optimizers.append(optimizer_adamw)

    param_groups = {
        "muon": muon_names,
        "adamw": adamw_names,
    }

    return optimizers, param_groups
