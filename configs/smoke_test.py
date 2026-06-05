"""Smoke test configuration for the 115M parameter Aero-Deuce variant.

This creates a scaled-down model (~115M params) for rapid validation on a single A10G GPU
using the TinyStories dataset. The architecture proportions are preserved from the full model.
"""

from configs.base import DataConfig, ModelConfig, TrainConfig


def smoke_test_model_config() -> ModelConfig:
    """115M parameter model configuration.

    Key scaling decisions:
    - d_model: 512 (4x reduction from 2048)
    - n_layers: 16 (2.25x reduction from 36)
    - n_experts: 16 (4x reduction from 64), top_k: 4 (from 6)
    - n_q_heads: 8, n_kv_heads: 4 (maintains 2:1 GQA ratio)
    - 3 attention layers at positions 4, 9, 14 (maintains ~3:1 SSM:Attn ratio)
    - Layers 0-2 still use dense FFN; layers 3-15 use MoE
    """
    return ModelConfig(
        # Core dimensions
        d_model=512,
        n_layers=16,
        vocab_size=151_936,
        tie_embeddings=True,
        use_bias=False,
        # Mamba-3 SSM
        ssm_d_state=64,
        ssm_expand=2,
        ssm_headdim=64,
        # GQA Attention
        n_q_heads=8,
        n_kv_heads=4,
        attn_theta=1_000_000.0,
        max_seq_len=4_096,
        # Layer assignments
        attn_layer_indices=(4, 9, 14),
        dense_ffn_layers=(0, 1, 2),
        # DeepSeekMoE
        n_experts=16,
        n_shared_experts=1,
        top_k=4,
        expert_hidden_dim=768,  # Fine-grained: 3/8 of standard 2048
        router_aux_loss_alpha=0.01,
        router_init_std=1e-3,
        # Init
        init_std=0.02,
    )


def smoke_test_train_config() -> TrainConfig:
    """Training configuration for smoke test on A10G (24GB VRAM)."""
    return TrainConfig(
        batch_size=4,
        micro_batch_size=1,
        max_seq_len=4_096,
        max_steps=10_000,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=500,
        grad_clip_norm=1.0,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_lr_scale=0.02,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        use_bf16=True,
        log_interval=10,
        eval_interval=1_000,
        checkpoint_interval=2_000,
        wandb_project="aero-deuce",
        wandb_run_name="smoke-test-115M",
        checkpoint_dir="/checkpoints",
    )


def smoke_test_data_config() -> DataConfig:
    """Data configuration for TinyStories smoke test."""
    return DataConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="train",
        dataset_text_field="text",
        tokenizer_name="Qwen/Qwen3-1.7B",
        eos_token_id=151645,
        streaming=True,
        pack_sequences=True,
        buffer_size=10_000,
        max_seq_len=4_096,
        num_workers=0,
        pin_memory=True,
    )
