"""Modal deployment for Aero-Deuce smoke test.

Runs the 115M parameter smoke test on an NVIDIA A10G GPU instance using Modal.

Usage:
    # Step 1: Validate model builds and one forward pass works
    python -m modal run aero_deuce/infra/modal_app.py

    # Step 2: Run the full smoke test training loop
    python -m modal run aero_deuce/infra/modal_app.py --train

Target: Loss should drop smoothly and bottom out below 1.1 on TinyStories.
If it plateaus at 2.5+, pause and audit the Mamba-3 kernels or routing logic.
"""

import modal

app = modal.App("aero-deuce")

# Persistent volume for checkpoints across runs
volume = modal.Volume.from_name("aero-deuce-checkpoints", create_if_missing=True)

# Container image: PyTorch devel image with CUDA toolkit (nvcc) for mamba-ssm kernel compilation.
# Project code is added via add_local_dir (mounted at container startup, not baked into image).
#
# Critical: mamba-ssm's dependency chain tries to upgrade torch to 2.12+ which breaks
# the CUDA kernel ABI. We pin torch and use --no-deps to prevent this.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    # Step 1: Upgrade to torch 2.7.0 — required for mamba-ssm 2.3.x's C++ ABI.
    # The devel image provides nvcc (CUDA 12.4); torch 2.7.0's pip wheel bundles
    # a compatible CUDA runtime. Build CUDA extensions with system nvcc, link against
    # torch 2.7.0's C++ libs.
    .pip_install("torch==2.7.0")
    # Step 2: Install mamba-ssm build-time deps
    .pip_install("ninja", "packaging", "einops", "setuptools", "triton>=3.5.0")
    # Step 3: Build CUDA packages from source against torch 2.7.0.
    # --no-build-isolation: uses system torch during compilation
    # --no-deps: prevents pip from overwriting our torch version
    .pip_install(
        "causal-conv1d>=1.6.0",
        extra_options="--no-build-isolation --no-deps",
    )
    .pip_install(
        "mamba-ssm>=2.3.0",
        extra_options="--no-build-isolation --no-deps",
    )
    # Mamba-3 MIMO kernel runtime deps (skipped by --no-deps above).
    # z3-solver provides libz3.so required by tilelang's bundled TVM.
    # cloudpickle, ml-dtypes, torch-c-dlpack-ext are tilelang runtime deps.
    .pip_install(
        "tilelang==0.1.8",
        "apache-tvm-ffi<=0.1.9",
        "quack-kernels>=0.3.4",
        "z3-solver>=4.13.0",
        "cloudpickle",
        "ml-dtypes",
        "torch-c-dlpack-ext",
        extra_options="--no-deps",
    )
    # Step 4: Install remaining runtime deps.
    # Pin transformers<5 — mamba-ssm 2.2.5 imports GreedySearchDecoderOnlyOutput
    # which was removed in transformers 5.x
    .pip_install(
        "transformers>=4.40.0,<5.0.0",
        "datasets>=2.18.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(
        local_path=".",
        remote_path="/root/aero-deuce",
        ignore=["__pycache__", ".git", "*.pyc", ".venv"],
    )
)


@app.function(
    gpu="A10G",
    image=image,
    timeout=600,  # 10 min for validation
)
def validate():
    """Validate the model builds and one forward/backward pass works.

    This is a cheap, fast check before committing to a long training run.
    """
    import sys
    import torch

    sys.path.insert(0, "/root/aero-deuce")

    from configs.smoke_test import smoke_test_model_config
    from aero_deuce.model.transformer import AeroDeuceForCausalLM

    device = torch.device("cuda")
    print(f"[Validate] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Validate] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build model
    config = smoke_test_model_config()
    print(f"[Validate] Building model: d_model={config.d_model}, layers={config.n_layers}, "
          f"experts={config.n_experts}, top_k={config.top_k}")

    model = AeroDeuceForCausalLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Validate] Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Forward pass with dummy data
    B, L = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, L), device=device)
    labels = torch.randint(0, config.vocab_size, (B, L), device=device)

    print(f"[Validate] Forward pass: input_ids {input_ids.shape}...")
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]

    print(f"[Validate] Loss: {loss.item():.4f}")
    print(f"[Validate] Logits shape: {outputs['logits'].shape}")
    if outputs.get("aux_loss") is not None:
        print(f"[Validate] Aux loss: {outputs['aux_loss'].item():.6f}")

    # Backward pass
    print("[Validate] Backward pass...")
    loss.backward()

    # Check gradients exist
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    n_params = sum(1 for p in model.parameters())
    print(f"[Validate] Gradients: {n_grads}/{n_params} parameters have gradients")

    # Memory report
    alloc = torch.cuda.max_memory_allocated() / 1e9
    print(f"[Validate] Peak GPU memory: {alloc:.2f} GB")

    print("\n[Validate] ✓ All checks passed — model is ready for training")
    return {
        "total_params": total_params,
        "loss": loss.item(),
        "peak_memory_gb": alloc,
        "grads_ok": n_grads == n_params,
    }


@app.function(
    gpu="A10G",
    image=image,
    timeout=86400,  # 24 hours
    volumes={"/checkpoints": volume},
)
def train():
    """Run the full 115M parameter smoke test training on TinyStories."""
    import sys
    import torch

    sys.path.insert(0, "/root/aero-deuce")

    from configs.smoke_test import smoke_test_model_config, smoke_test_train_config, smoke_test_data_config
    from aero_deuce.model.transformer import AeroDeuceForCausalLM
    from aero_deuce.training.trainer import Trainer

    # Configs
    model_config = smoke_test_model_config()
    train_config = smoke_test_train_config()
    data_config = smoke_test_data_config()

    # Device
    device = torch.device("cuda")
    print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build model
    model = AeroDeuceForCausalLM(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Create trainer
    trainer = Trainer(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        model=model,
        device=device,
    )

    print(f"\n[Train] Starting training for {train_config.max_steps} steps")
    print(f"[Train] Seq length: {train_config.max_seq_len}, "
          f"Batch: {train_config.batch_size} (micro: {train_config.micro_batch_size})")
    print(f"[Train] LR: {train_config.learning_rate}, Warmup: {train_config.warmup_steps}")
    print()

    trainer.train()

    print(f"\n[Train] Complete at step {trainer.step}")

    # Persist checkpoints
    volume.commit()


@app.local_entrypoint()
def main(train: bool = False):
    """Entry point. Defaults to validation; pass --train for full training."""
    if train:
        print("=" * 60)
        print("  Aero-Deuce FULL SMOKE TEST")
        print("  115M params · TinyStories · A10G")
        print("=" * 60)
        train.remote()
    else:
        print("=" * 60)
        print("  Aero-Deuce VALIDATION")
        print("  Build model + 1 forward/backward pass")
        print("=" * 60)
        result = validate.remote()
        print(f"\nResult: {result}")
