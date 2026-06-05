"""Aero-Deuce configuration dataclasses.

Three frozen dataclasses that fully parameterize the model, training, and data pipeline.
These are the single source of truth — every module imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for Aero-Deuce."""

    # Core dimensions
    d_model: int = 2048
    n_layers: int = 36
    vocab_size: int = 151_936  # Qwen3 tokenizer
    tie_embeddings: bool = True
    use_bias: bool = False

    # Mamba-3 SSM parameters
    ssm_d_state: int = 128
    ssm_expand: int = 2
    ssm_headdim: int = 64

    # Grouped-Query Attention parameters
    n_q_heads: int = 16
    n_kv_heads: int = 8
    attn_theta: float = 1_000_000.0  # RoPE base frequency
    max_seq_len: int = 32_768

    # Layer type assignments (0-indexed)
    attn_layer_indices: Tuple[int, ...] = (9, 21, 33)
    dense_ffn_layers: Tuple[int, ...] = (0, 1, 2)

    # DeepSeekMoE parameters
    n_experts: int = 64
    n_shared_experts: int = 1
    top_k: int = 6
    expert_hidden_dim: int = 0  # 0 = auto-compute (4 * d_model * 7 / (64 * n_shared_experts))
    router_aux_loss_alpha: float = 0.01
    router_init_std: float = 1e-3

    # RMSNorm epsilon
    rms_norm_eps: float = 1e-6

    # Weight initialization
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_q_heads

    @property
    def computed_expert_hidden_dim(self) -> int:
        """Compute per-expert hidden dimension following DeepSeekMoE fine-grained segmentation.

        The standard SwiGLU FFN has hidden_dim = 4 * d_model.
        With N routed experts, each expert gets a fraction to keep total FLOPs comparable.
        We use 7/64 of the standard hidden dim per expert (matching DeepSeek-MoE proportions).
        """
        if self.expert_hidden_dim > 0:
            return self.expert_hidden_dim
        # Standard FFN hidden, then scale for fine-grained experts
        standard_hidden = 4 * self.d_model
        # DeepSeek proportion: each expert is ~7/64 of standard, rounded to nearest 64 for alignment
        per_expert = max(64, (standard_hidden * 7 // (64 * self.n_shared_experts) // 64) * 64)
        return per_expert

    @property
    def ffn_hidden_dim(self) -> int:
        """Standard (dense) FFN hidden dimension."""
        return 4 * self.d_model


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    # Batch configuration
    batch_size: int = 4  # Global batch size (per optimizer step)
    micro_batch_size: int = 1  # Per-forward-pass micro batch (gradient accumulation)

    # Sequence length (overrides ModelConfig.max_seq_len for data pipeline)
    max_seq_len: int = 4_096

    # Optimization
    max_steps: int = 100_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    grad_clip_norm: float = 1.0

    # Muon optimizer (for 2D weight matrices)
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5  # Newton-Schulz iterations
    muon_lr_scale: float = 0.02  # Muon effective LR = lr * muon_lr_scale

    # AdamW optimizer (for embeddings, norms, router)
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8

    # Precision
    use_bf16: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 1_000
    checkpoint_interval: int = 5_000
    wandb_project: str = "aero-deuce"
    wandb_run_name: str = ""

    # Checkpointing
    checkpoint_dir: str = "/checkpoints"

    @property
    def grad_accum_steps(self) -> int:
        return self.batch_size // self.micro_batch_size


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration."""

    # Dataset
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    dataset_text_field: str = "text"

    # Tokenizer
    tokenizer_name: str = "Qwen/Qwen3-1.7B"
    eos_token_id: int = 151645  # Qwen3 <|endoftext|>

    # Streaming and packing
    streaming: bool = True
    pack_sequences: bool = True
    buffer_size: int = 10_000  # Streaming shuffle buffer

    # Sequence configuration
    max_seq_len: int = 4_096

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True
