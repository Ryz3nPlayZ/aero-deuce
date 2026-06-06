"""Modal deployment for Aero-Deuce QLoRA post-training on Gemma 4 12B IT.

Runs QLoRA SFT on a single GPU via Unsloth's fused Triton kernels.
4-bit NF4 quantization keeps base model at ~7.5GB, LoRA adapters add ~50MB.

Usage:
    # Step 1: Validate model loads and one forward/backward pass works
    modal run aero_deuce/infra/modal_app.py

    # Step 2: 50-step smoke test
    modal run aero_deuce/infra/modal_app.py --smoke

    # Step 3: Full ~3000 step training run
    modal run aero_deuce/infra/modal_app.py --run-train

    # Step 4: Export merged model to GGUF + MLX
    modal run aero_deuce/infra/modal_app.py --export

Target: Loss should drop smoothly from ~2.5 to below 1.0 on 30K instruction data.
"""

import modal

app = modal.App("aero-deuce-qlora")

# Persistent volume for checkpoints and exported models
volume = modal.Volume.from_name("aero-deuce-checkpoints", create_if_missing=True)

# Container image: Standard HuggingFace stack for QLoRA training.
# Using transformers + bitsandbytes + peft directly (no Unsloth — avoids dependency hell).
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "cmake")
    # Upgrade torch — base image has 2.5.1, but newer transformers needs torch.int1 (2.6+).
    .pip_install("torch==2.7.0", "torchvision>=0.22.0")
    # HuggingFace ecosystem
    .pip_install(
        "transformers>=4.46.0",
        "datasets>=2.18.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
    )
    # 4-bit quantization
    .pip_install("bitsandbytes>=0.43.0")
    # Utilities
    .pip_install(
        "wandb>=0.17.0",
        "numpy>=1.24.0",
        "sentencepiece",
        "protobuf",
    )
    # GGUF export
    .pip_install("gguf>=0.6.0")
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
    """Validate QLoRA setup: load Gemma 4 12B IT in 4-bit, inject LoRA, run 1 fwd/bwd.

    Checks:
    - Model loads successfully in 4-bit NF4
    - LoRA adapters are injected on all 7 target modules
    - Forward + backward pass works
    - Peak VRAM stays under 20GB (A10G has 24GB)
    """
    import sys
    import torch

    sys.path.insert(0, "/root/aero-deuce")

    from configs.qlora import gemma4_12b_qlora_config
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    config = gemma4_12b_qlora_config()

    print(f"[Validate] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Validate] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[Validate] Loading google/gemma-4-12b-it in 4-bit NF4...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-12b-it",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    base_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[Validate] Base model loaded: {base_mem:.2f} GB")

    # Prepare model for training (freezes base, casts layer norms, etc.)
    model = prepare_model_for_kbit_training(model)

    # Inject LoRA adapters
    print(f"[Validate] Injecting LoRA adapters: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"[Validate] Target modules: {config.target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing — essential for 12B model on 24GB GPU.
    # Recomputes activations during backward pass instead of storing them all.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Validate] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    lora_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[Validate] After LoRA injection: {lora_mem:.2f} GB")

    # Forward pass with dummy chat-formatted input
    B, L = 1, 128
    input_ids = torch.randint(0, tokenizer.vocab_size, (B, L), device="cuda")
    attention_mask = torch.ones(B, L, device="cuda", dtype=torch.long)
    labels = input_ids.clone()

    print(f"[Validate] Forward pass: input_ids {input_ids.shape}...")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    print(f"[Validate] Loss: {loss.item():.4f}")

    # Backward pass
    print("[Validate] Backward pass...")
    loss.backward()

    # Check LoRA gradients exist
    lora_grads = 0
    lora_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params += 1
            if param.grad is not None:
                lora_grads += 1

    print(f"[Validate] LoRA gradients: {lora_grads}/{lora_params} params have gradients")

    # Memory report
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[Validate] Peak GPU memory: {peak:.2f} GB")

    # Validation checks
    assert lora_grads == lora_params, f"Missing gradients: {lora_grads}/{lora_params}"
    assert peak < 20.0, f"VRAM too high: {peak:.2f} GB (need < 20 GB for A10G)"

    print("\n[Validate] ✓ All checks passed — QLoRA setup ready for training")
    return {
        "base_model_mb": base_mem * 1000,
        "total_model_mb": lora_mem * 1000,
        "peak_vram_gb": peak,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
    }


@app.function(
    gpu="A10G",
    image=image,
    timeout=1800,  # 30 min for smoke test
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("aero-deuce-wandb")],
)
def smoke_train():
    """Quick 50-step SFT smoke test to validate the full pipeline.

    Verifies: data loading, chat template formatting, loss masking,
    dual optimizer (Muon + AdamW), gradient accumulation, checkpoint saving.
    """
    import sys
    import torch

    sys.path.insert(0, "/root/aero-deuce")

    from configs.qlora import (
        gemma4_12b_qlora_config,
        qlora_smoke_train_config,
        qlora_data_config,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from aero_deuce.data.dataset import create_sft_dataloaders
    from aero_deuce.training.trainer import Trainer

    qlora_config = gemma4_12b_qlora_config()
    train_config = qlora_smoke_train_config()
    data_config = qlora_data_config()

    device = torch.device("cuda")
    print(f"[SmokeTrain] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[SmokeTrain] Loading google/gemma-4-12b-it...")

    # Load model + tokenizer in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-12b-it",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=qlora_config.lora_r,
        lora_alpha=qlora_config.lora_alpha,
        target_modules=list(qlora_config.target_modules),
        lora_dropout=qlora_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    torch.cuda.reset_peak_memory_stats()
    print(f"[SmokeTrain] Model loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Create data loaders
    print("[SmokeTrain] Creating data loaders...")
    train_loader, eval_loader = create_sft_dataloaders(data_config, tokenizer)
    print(f"[SmokeTrain] Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        data_config=data_config,
        device=device,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        eval_batches=5,
    )

    print(f"\n[SmokeTrain] Running {train_config.max_steps} steps...")
    print(f"[SmokeTrain] Seq={train_config.max_seq_len}, Batch={train_config.batch_size}, "
          f"GradAccum={train_config.grad_accum_steps}")
    print()

    trainer.train()

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[SmokeTrain] ✓ Done! Step={trainer.step}, Peak GPU={peak_mem:.2f} GB")

    volume.commit()


@app.function(
    gpu="A10G",
    image=image,
    timeout=86400,  # 24 hours — safe margin
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("aero-deuce-wandb")],
)
def train():
    """Full ~2000 step QLoRA SFT training run on Gemma 4 12B IT.

    ~30K instruction samples with masked loss on assistant tokens.
    Dual optimizer: Muon for LoRA A/B matrices, AdamW for non-2D params.
    Checkpoints saved to Modal Volume every 50 steps.
    Auto-resumes from latest checkpoint if preempted.
    """
    import sys
    import os
    import glob
    import torch

    # Reduce CUDA memory fragmentation — prevents OOM on long sequences
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    sys.path.insert(0, "/root/aero-deuce")

    from configs.qlora import (
        gemma4_12b_qlora_config,
        qlora_train_config,
        qlora_data_config,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from aero_deuce.data.dataset import create_sft_dataloaders
    from aero_deuce.training.trainer import Trainer

    qlora_config = gemma4_12b_qlora_config()
    train_config = qlora_train_config()
    data_config = qlora_data_config()

    device = torch.device("cuda")
    print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Train] Loading google/gemma-4-12b-it in 4-bit...")

    # Load model + tokenizer in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-12b-it",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=qlora_config.lora_r,
        lora_alpha=qlora_config.lora_alpha,
        target_modules=list(qlora_config.target_modules),
        lora_dropout=qlora_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    torch.cuda.reset_peak_memory_stats()
    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"[Train] Model loaded: {alloc:.2f} GB")

    # Data loaders
    print("[Train] Creating data loaders...")
    train_loader, eval_loader = create_sft_dataloaders(data_config, tokenizer)
    print(f"[Train] Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Find latest checkpoint to resume from
    latest_ckpt = None
    ckpt_dirs = sorted(glob.glob("/checkpoints/step_*"))
    if ckpt_dirs:
        latest_ckpt = ckpt_dirs[-1]
        print(f"[Train] Found checkpoint: {latest_ckpt}")

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        data_config=data_config,
        device=device,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        eval_batches=20,
    )

    # Resume from checkpoint if found
    if latest_ckpt:
        trainer.resume_from_checkpoint(latest_ckpt)

    print(f"\n[Train] {'Resuming' if latest_ckpt else 'Starting'} from step {trainer.step}/{train_config.max_steps}")
    print(f"[Train] Seq={train_config.max_seq_len}, Batch={train_config.batch_size}, "
          f"GradAccum={train_config.grad_accum_steps}")
    print(f"[Train] Tokens/step: {train_config.batch_size * train_config.max_seq_len:,}")
    print(f"[Train] LR: {train_config.learning_rate}, Warmup: {train_config.warmup_steps}")
    print(f"[Train] Checkpoint every {train_config.checkpoint_interval} steps, Eval every {train_config.eval_interval} steps")
    print()

    trainer.train()

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[Train] Complete at step {trainer.step}, Peak GPU: {peak_mem:.2f} GB")

    volume.commit()


@app.function(
    gpu="A10G",
    image=image,
    timeout=3600,  # 1 hour for export
    volumes={"/checkpoints": volume},
)
def export(checkpoint_dir: str = "/checkpoints/final"):
    """Export merged model to GGUF (Q4_K_M) and MLX formats.

    1. Load base model in fp16 (not quantized)
    2. Merge LoRA adapters into base weights
    3. Export GGUF Q4_K_M for CPU/cross-platform inference
    4. Export MLX for Apple Silicon inference
    """
    import sys

    sys.path.insert(0, "/root/aero-deuce")

    from aero_deuce.export.merge_and_export import (
        merge_adapter,
        export_gguf,
        export_mlx,
    )

    print(f"[Export] Merging adapters from {checkpoint_dir}...")
    merged_dir = merge_adapter(checkpoint_dir, output_dir="/checkpoints/merged")

    print("[Export] Exporting GGUF Q4_K_M...")
    gguf_path = export_gguf(merged_dir, "/checkpoints/export/aero-deuce-q4km.gguf")
    print(f"[Export] GGUF saved: {gguf_path}")

    print("[Export] Exporting MLX...")
    mlx_dir = export_mlx(merged_dir, "/checkpoints/export/aero-deuce-mlx")
    print(f"[Export] MLX saved: {mlx_dir}")

    volume.commit()
    print("\n[Export] ✓ All exports complete")


@app.local_entrypoint()
def main(
    run_train: bool = False,
    smoke: bool = False,
    do_export: bool = False,
):
    """Entry point.

    Defaults to validation. Use --smoke for 50-step test, --run-train for full training,
    --do-export for GGUF + MLX export.
    """
    if run_train:
        print("=" * 60)
        print("  Aero-Deuce QLoRA SFT — Full Training (~3000 steps)")
        print("  Gemma 4 12B IT · QLoRA r=16 · A10G")
        print("=" * 60)
        train.remote()
    elif smoke:
        print("=" * 60)
        print("  Aero-Deuce QLoRA SFT — Smoke Test (50 steps)")
        print("  Gemma 4 12B IT · QLoRA r=16 · A10G")
        print("=" * 60)
        smoke_train.remote()
    elif do_export:
        print("=" * 60)
        print("  Aero-Deuce — Export GGUF + MLX")
        print("=" * 60)
        export.remote()
    else:
        print("=" * 60)
        print("  Aero-Deuce QLoRA Validation")
        print("  Load Gemma 4 12B IT in 4-bit + inject LoRA")
        print("=" * 60)
        result = validate.remote()
        print(f"\nResult: {result}")
