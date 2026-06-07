"""Aero-Deuce QLoRA SFT training script for Lightning AI.

Resumes from a checkpoint saved on Modal and completes the remaining training.
Runs directly on the Lightning AI studio machine — no container orchestration needed.

Usage:
    python scripts/lightning_train.py

Environment:
    - A100 80GB (or any CUDA GPU with 24GB+ VRAM)
    - Dependencies: torch, transformers, peft, bitsandbytes, datasets, accelerate
    - Checkpoint at ./checkpoints/step_1000/ (adapter_model.safetensors + training_state.pt)
"""

import os
import sys
import glob
import torch

# Reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.qlora import (
    gemma4_12b_qlora_config,
    qlora_train_config,
    qlora_data_config,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from aero_deuce.data.dataset import create_sft_dataloaders
from aero_deuce.training.trainer import Trainer


def main():
    qlora_config = gemma4_12b_qlora_config()
    train_config = qlora_train_config()
    data_config = qlora_data_config()

    # Override checkpoint dir to local project path (not /checkpoints which is Modal-specific)
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    from dataclasses import replace
    train_config = replace(train_config, checkpoint_dir=checkpoint_dir)

    device = torch.device("cuda")
    print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    latest_ckpt = None
    ckpt_dirs = sorted(
        glob.glob(os.path.join(checkpoint_dir, "step_*")),
        key=lambda p: int(p.split("step_")[-1]),
    )
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


if __name__ == "__main__":
    main()
