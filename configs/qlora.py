"""QLoRA configuration factory for Gemma 4 12B IT post-training.

Provides pre-built config instances with sensible defaults for each training stage.
"""

from configs.base import QLoRAConfig, TrainConfig, DataConfig


def gemma4_12b_qlora_config() -> QLoRAConfig:
    """Standard QLoRA config for Gemma 4 12B IT.

    4-bit NF4 with double quantization, LoRA r=16/alpha=32 on all linear layers.
    Peaks at ~7.5GB base model + ~50MB LoRA adapters = ~7.6GB model weight VRAM.
    """
    return QLoRAConfig()


def qlora_train_config() -> TrainConfig:
    """Training config for full QLoRA SFT run on A10G (24GB VRAM).

    ~1500 steps × 4 batch × 2048 seq with grad checkpointing.
    At ~265 tok/s (observed), this takes ~8 hours (~$5 on spot).
    """
    return TrainConfig(
        batch_size=4,
        micro_batch_size=1,
        max_seq_len=2048,
        max_steps=1_500,
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
    """Quick 50-step smoke test to validate the full SFT pipeline.

    With seq_len=2048 and grad checkpointing, ~30 sec/step → ~25 min.
    """
    return TrainConfig(
        batch_size=4,
        micro_batch_size=1,
        max_seq_len=2048,
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
