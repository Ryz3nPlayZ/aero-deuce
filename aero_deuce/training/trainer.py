"""SFT Training loop for Aero-Deuce QLoRA post-training.

Handles gradient accumulation, dual optimizer stepping (Muon + AdamW),
gradient clipping, wandb logging, learning rate scheduling, periodic evaluation,
and LoRA adapter checkpointing.

Adapted from the pretraining trainer to work with HuggingFace model API
(outputs.loss instead of custom dict) and save LoRA adapter weights only.
"""

import time
from contextlib import nullcontext
from pathlib import Path

import torch

from configs.base import TrainConfig, DataConfig
from aero_deuce.optim.dual_optim import create_dual_optimizer
from aero_deuce.training.schedule import get_lr


class Trainer:
    """Orchestrates the QLoRA SFT training loop on Gemma 4 12B IT.

    Supports:
    - Gradient accumulation across micro-batches
    - Dual optimizer (Muon for LoRA A/B, AdamW for non-2D params)
    - Warmup + cosine decay LR schedule applied to both optimizers
    - Gradient clipping by global norm
    - Periodic evaluation on held-out data
    - wandb logging for loss, LR, grad norm, throughput
    - LoRA adapter-only checkpointing (much smaller than full model saves)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        train_config: TrainConfig,
        data_config: DataConfig,
        device: torch.device,
        train_dataloader=None,
        eval_dataloader=None,
        eval_batches: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.data_config = data_config
        self.device = device
        self.eval_dataloader = eval_dataloader
        self.eval_batches = eval_batches

        # Create dual optimizer (Muon + AdamW)
        self.optimizers, self.param_groups = create_dual_optimizer(model, train_config)

        # Use provided dataloader or create one
        self.dataloader = train_dataloader

        # Training state
        self.step = 0
        self.best_loss = float("inf")

    def train(self) -> None:
        """Run the full training loop."""
        if self.dataloader is None:
            raise ValueError("No train dataloader provided. Pass train_dataloader to constructor.")

        self.model.train()

        # Initialize wandb (graceful fallback if no API key)
        use_wandb = False
        try:
            import wandb
            wandb.init(
                project=self.train_config.wandb_project,
                name=self.train_config.wandb_run_name,
                config={
                    "train": vars(self.train_config),
                    "data": vars(self.data_config),
                },
            )
            use_wandb = True
        except Exception as e:
            print(f"[Trainer] wandb not available ({type(e).__name__}), logging to stdout only")

        data_iter = iter(self.dataloader)
        step_time = time.time()

        # bf16 autocast context
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        while self.step < self.train_config.max_steps:
            accumulated_loss = 0.0

            for _ in range(self.train_config.grad_accum_steps):
                # Get next batch (recreate iterator if exhausted)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass (with bf16 autocast)
                with amp_ctx if self.train_config.use_bf16 else nullcontext():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.train_config.grad_accum_steps

                # Backward pass
                loss.backward()
                accumulated_loss += loss.item()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.grad_clip_norm,
            )

            # Update learning rates
            lr = get_lr(
                step=self.step,
                max_steps=self.train_config.max_steps,
                warmup_steps=self.train_config.warmup_steps,
                lr=self.train_config.learning_rate,
            )
            muon_lr = lr * self.train_config.muon_lr_scale

            # Apply LR to optimizers based on which ones exist
            # optimizers[0] = Muon (lr scaled), optimizers[1] = AdamW (base lr)
            # But we can't rely on ordering — use param_groups to identify
            for opt_idx, opt in enumerate(self.optimizers):
                target_lr = muon_lr if opt_idx == 0 else lr
                for group in opt.param_groups:
                    group["lr"] = target_lr

            # Step both optimizers
            for opt in self.optimizers:
                opt.step()

            # Zero gradients
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            self.step += 1

            # Compute throughput
            now = time.time()
            step_time_ms = (now - step_time) * 1000
            tokens_per_sec = (
                self.train_config.micro_batch_size
                * self.data_config.max_seq_len
                * self.train_config.grad_accum_steps
                / (now - step_time)
            )
            step_time = now

            # Logging
            if self.step % self.train_config.log_interval == 0:
                log_data = {
                    "train/loss": accumulated_loss,
                    "train/lr": lr,
                    "train/muon_lr": muon_lr,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/step_time_ms": step_time_ms,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": self.step,
                }

                if use_wandb:
                    import wandb
                    wandb.log(log_data, step=self.step)

                # Console logging
                print(
                    f"Step {self.step:>6d} | "
                    f"Loss {accumulated_loss:.4f} | "
                    f"LR {lr:.2e} | "
                    f"GradNorm {grad_norm:.4f} | "
                    f"{tokens_per_sec:.0f} tok/s | "
                    f"{step_time_ms:.0f} ms/step"
                )

            # Evaluation
            if (
                self.train_config.eval_interval > 0
                and self.step % self.train_config.eval_interval == 0
            ):
                val_loss = self.evaluate()
                print(
                    f"  └─ Eval @ step {self.step}: "
                    f"val_loss={val_loss:.4f}, "
                    f"train_val_gap={accumulated_loss - val_loss:+.4f}"
                )

                if use_wandb:
                    import wandb
                    wandb.log({
                        "eval/loss": val_loss,
                        "eval/train_val_gap": accumulated_loss - val_loss,
                        "eval/step": self.step,
                    }, step=self.step)

            # Checkpointing
            if self.step % self.train_config.checkpoint_interval == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()
        if use_wandb:
            import wandb
            wandb.finish()

    def evaluate(self) -> float:
        """Run evaluation on the held-out split.

        Runs self.eval_batches forward passes in no_grad mode and returns
        the average loss.

        Returns:
            Average validation loss.
        """
        if self.eval_dataloader is None:
            print("[Eval] No eval dataloader configured, skipping evaluation")
            return 0.0

        self.model.eval()
        eval_iter = iter(self.eval_dataloader)
        total_loss = 0.0
        n_batches = 0

        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            for _ in range(self.eval_batches):
                try:
                    batch = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(self.eval_dataloader)
                    batch = next(eval_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with amp_ctx if self.train_config.use_bf16 else nullcontext():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                total_loss += outputs.loss.item()
                n_batches += 1

        self.model.train()

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def save_checkpoint(self) -> None:
        """Save LoRA adapter checkpoint.

        Saves only the LoRA adapter weights (not the full 12B base model)
        plus optimizer states and training step. This is much more efficient
        than saving the full model.
        """
        checkpoint_dir = Path(self.train_config.checkpoint_dir) / f"step_{self.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights via PEFT's save_pretrained
        self.model.save_pretrained(str(checkpoint_dir))

        # Save optimizer states and step counter separately
        torch.save({
            "step": self.step,
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "train_config": vars(self.train_config),
        }, checkpoint_dir / "training_state.pt")

        print(f"[Checkpoint] Saved LoRA adapter to {checkpoint_dir}")
