"""Modal deployment for Aero-Deuce smoke test.

Runs the 115M parameter smoke test on an NVIDIA A10G GPU instance using Modal.
Everything is code — no YAML configuration needed.

Usage:
    modal run aero_deuce/infra/modal_app.py

The training loop:
1. Builds the 115M model with the smoke test configuration
2. Streams TinyStories via HuggingFace datasets
3. Packs tokens into fixed-length sequences (4096)
4. Trains with dual Muon + AdamW optimizers
5. Logs loss curves to wandb
6. Saves checkpoints to a persistent Modal Volume

Target: Loss should drop smoothly and bottom out below 1.1 on TinyStories.
If it plateaus at 2.5+, pause and audit the Mamba-3 kernels or routing logic.
"""

import modal

app = modal.App("aero-deuce")

# Persistent volume for checkpoints across runs
volume = modal.Volume.from_name("aero-deuce-checkpoints", create_if_missing=True)

# Container image with CUDA dependencies for mamba-ssm
# Fallback: if mamba-ssm wheel fails on debian_slim, use a CUDA base image:
#   modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "nvidia-cuda-toolkit")
    .uv_pip_install(
        "torch>=2.4.0",
        "mamba-ssm>=2.2.0",
        "causal-conv1d>=1.4.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "wandb>=0.17.0",
        "numpy>=1.24.0",
    )
)


@app.function(
    gpu="A10G",
    image=image,
    timeout=86400,  # 24 hours max
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("wandb-api-key", required_keys=["WANDB_API_KEY"])],
)
def run_smoke_test():
    """Run the 115M parameter smoke test on TinyStories."""
    import torch
    import sys

    # Add project root to path for imports
    project_root = "/root/aero-deuce"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from configs.smoke_test import smoke_test_model_config, smoke_test_train_config, smoke_test_data_config
    from aero_deuce.model.transformer import AeroDeuceForCausalLM
    from aero_deuce.training.trainer import Trainer

    # Configs
    model_config = smoke_test_model_config()
    train_config = smoke_test_train_config()
    data_config = smoke_test_data_config()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Modal] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Modal] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Modal] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Build model
    print(f"[Modal] Building {model_config.n_layers}-layer model (d_model={model_config.d_model})...")
    model = AeroDeuceForCausalLM(model_config)
    model = model.to(device)

    # Report parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modal] Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"[Modal] Trainable parameters: {trainable_params:,}")

    # Create trainer and run
    trainer = Trainer(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        model=model,
        device=device,
    )

    print(f"[Modal] Starting training for {train_config.max_steps} steps...")
    print(f"[Modal] Sequence length: {train_config.max_seq_len}")
    print(f"[Modal] Batch size: {train_config.batch_size} (micro: {train_config.micro_batch_size})")
    print(f"[Modal] Learning rate: {train_config.learning_rate}")
    print()

    trainer.train()

    print(f"\n[Modal] Training complete at step {trainer.step}")
    print(f"[Modal] Best loss: {trainer.best_loss:.4f}")

    # Commit volume to persist checkpoints
    volume.commit()


@app.local_entrypoint()
def main():
    """Entry point — runs the smoke test remotely on Modal."""
    print("=" * 60)
    print("  Aero-Deuce Smoke Test")
    print("  115M params · TinyStories · A10G")
    print("=" * 60)
    print()
    print("Launching remote training job...")
    run_smoke_test.remote()
