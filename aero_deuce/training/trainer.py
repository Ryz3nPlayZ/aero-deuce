"""Training loop for Aero-Deuce.

Handles gradient accumulation, dual optimizer stepping, gradient clipping,
wandb logging, learning rate scheduling, and checkpointing.
"""

import time
from contextlib import nullcontext
from pathlib import Path

import torch

from configs.base import ModelConfig, TrainConfig, DataConfig
from aero_deuce.data.dataset import create_dataloader
from aero_deuce.optim.dual_optim import create_dual_optimizer
from aero_deuce.training.schedule import get_lr


class Trainer:
    """Orchestrates the Aero-Deuce training loop.

    Supports:
    - Gradient accumulation across micro-batches
    - Dual optimizer (Muon + AdamW) stepping
    - Warmup + cosine decay LR schedule applied to both optimizers
    - Gradient clipping by global norm
    - wandb logging for loss, LR, grad norm, throughput
    - Periodic checkpointing to disk (or Modal Volume)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        data_config: DataConfig,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.model = model
        self.device = device

        # Create dual optimizer
        self.optimizers, self.param_groups = create_dual_optimizer(model, train_config)

        # Create data loader
        self.dataloader = create_dataloader(data_config, batch_size=train_config.micro_batch_size)

        # Training state
        self.step = 0
        self.best_loss = float("inf")

    def train(self) -> None:
        """Run the full training loop."""
        self.model.train()

        # Initialize wandb (graceful fallback if no API key)
        use_wandb = False
        try:
            import wandb
            wandb.init(
                project=self.train_config.wandb_project,
                name=self.train_config.wandb_run_name,
                config={
                    "model": vars(self.model_config),
                    "train": vars(self.train_config),
                    "data": vars(self.data_config),
                },
            )
            use_wandb = True
        except Exception as e:
            print(f"[Trainer] wandb not available ({type(e).__name__}), logging to stdout only")

        data_iter = iter(self.dataloader)
        step_time = time.time()

        # bf16 autocast context — saves VRAM and speeds up matmuls on Ampere+
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        while self.step < self.train_config.max_steps:
            accumulated_loss = 0.0
            accumulated_aux_loss = 0.0

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

                # Forward pass (with bf16 autocast if enabled)
                with amp_ctx if self.train_config.use_bf16 else nullcontext():
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / self.train_config.grad_accum_steps
                    aux_loss = outputs.get("aux_loss", torch.tensor(0.0))

                # Backward pass (outside autocast for stable fp32 gradients)
                loss.backward()

                accumulated_loss += loss.item()
                if aux_loss is not None:
                    accumulated_aux_loss += aux_loss.item()

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

            # Apply LR to both optimizer groups
            for group in self.optimizers[0].param_groups:
                group["lr"] = muon_lr
            for group in self.optimizers[1].param_groups:
                group["lr"] = lr

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
                    "train/aux_loss": accumulated_aux_loss,
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
                    f"AuxLoss {accumulated_aux_loss:.4f} | "
                    f"LR {lr:.2e} | "
                    f"GradNorm {grad_norm:.4f} | "
                    f"{tokens_per_sec:.0f} tok/s | "
                    f"{step_time_ms:.0f} ms/step"
                )

            # Checkpointing
            if self.step % self.train_config.checkpoint_interval == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()
        if use_wandb:
            import wandb
            wandb.finish()

    def save_checkpoint(self) -> None:
        """Save training checkpoint.

        Saves model state dict, optimizer states, and training step.
        Compatible with Modal Volumes and local filesystem.
        """
        checkpoint_dir = Path(self.train_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.step}.pt"

        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "model_config": vars(self.model_config),
            "train_config": vars(self.train_config),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[Checkpoint] Saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint and restore state.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        for opt, opt_state in zip(self.optimizers, checkpoint["optimizers"]):
            opt.load_state_dict(opt_state)
        self.step = checkpoint["step"]

        print(f"[Checkpoint] Restored from {checkpoint_path} at step {self.step}")
