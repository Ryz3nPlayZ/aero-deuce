"""QLoRA configuration factory for Gemma 4 12B IT post-training.

Provides pre-built config instances with sensible defaults for each training stage.
"""

from configs.base import QLoRAConfig, TrainConfig, DataConfig


def gemma4_12b_qlora_config() -> QLoRAConfig:
    """Standard QLoRA config for Gemma 4 12B IT."""
    return QLoRAConfig()


def qlora_train_config() -> TrainConfig:
    """Training config for full QLoRA SFT run on A10G (24GB VRAM).

    seq_len=1024, batch_size=2 with grad checkpointing to avoid OOM.
    ~2000 steps. At ~30s/step = ~17 hours (~$10 on spot).
    """
    return TrainConfig(
        batch_size=2,
        micro_batch_size=1,
        max_seq_len=1024,
        max_steps=2_000,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=50,
        grad_clip_norm=1.0,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_lr_scale=0.02,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        use_bf16=True,
        log_interval=10,
        eval_interval=250,
        checkpoint_interval=250,
        wandb_project="aero-deuce-qlora",
        wandb_run_name="gemma4-12b-qlora-sft",
        checkpoint_dir="/checkpoints",
    )


def qlora_smoke_train_config() -> TrainConfig:
    """Quick 50-step smoke test."""
    return TrainConfig(
        batch_size=2,
        micro_batch_size=1,
        max_seq_len=1024,
        max_steps=50,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=5,
        grad_clip_norm=1.0,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_lr_scale=0.02,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        use_bf16=True,
        log_interval=1,
        eval_interval=50,
        checkpoint_interval=50,
        wandb_project="aero-deuce-qlora",
        wandb_run_name="gemma4-12b-qlora-smoke",
        checkpoint_dir="/checkpoints",
    )


def qlora_data_config() -> DataConfig:
    """Standard data config: 30K samples, instruction + conversation mix."""
    return DataConfig()
