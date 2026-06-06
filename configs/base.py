"""Aero-Deuce configuration dataclasses.

Three frozen dataclasses that fully parameterize the QLoRA post-training pipeline
on Gemma 4 12B IT. These are the single source of truth — every module imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class QLoRAConfig:
    """QLoRA adapter configuration for Gemma 4 12B IT.

    Controls model loading (4-bit NF4 quantization) and LoRA adapter injection.
    Uses Unsloth's FastLanguageModel for fused Triton kernels.
    """

    # Base model — IT variant has instruction following + thinking mode baked in
    base_model: str = "google/gemma-4-12b-it"

    # Sequence length (truncation/padding target)
    max_seq_length: int = 1024              # Reduced from 2048 — OOM protection

    # 4-bit quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"          # NormalFloat4
    bnb_4bit_use_double_quant: bool = True     # Nested quantization for further compression
    bnb_4bit_compute_dtype: str = "bfloat16"   # Compute dtype for dequantized weights

    # LoRA hyperparameters
    lora_r: int = 16                           # Rank
    lora_alpha: int = 32                       # Scaling factor (alpha/r = 2.0)
    lora_dropout: float = 0.0                  # No dropout — dataset is small, want full signal
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Gradient checkpointing to save VRAM
    use_gradient_checkpointing: bool = True

    # dtype for model loading — None = auto-detect (bf16 on Ampere+, fp16 otherwise)
    dtype: str = None


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters for QLoRA SFT.

    Tuned for ~30K sample dataset on a single A10G (24GB VRAM).
    ~3000 steps at batch_size=4, grad_accum=4 → ~48K gradient updates total.
    """

    # Batch configuration
    batch_size: int = 2                        # Reduced from 4 — OOM protection on 24GB GPU
    micro_batch_size: int = 1                  # Per-forward-pass (gradient accumulation)

    max_seq_len: int = 1024                  # Reduced from 2048 — OOM protection

    # Optimization
    max_steps: int = 2_000                   # More steps to compensate for smaller batch
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip_norm: float = 1.0

    # Muon optimizer (for 2D LoRA weight matrices: lora_A, lora_B)
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5                     # Newton-Schulz iterations
    muon_lr_scale: float = 0.02                # Muon effective LR = lr * muon_lr_scale

    # AdamW optimizer (for non-2D trainable params: scalars, embedding LoRA)
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8

    # Precision
    use_bf16: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 500
    wandb_project: str = "aero-deuce-qlora"
    wandb_run_name: str = ""

    # Checkpointing
    checkpoint_dir: str = "/checkpoints"

    @property
    def grad_accum_steps(self) -> int:
        return self.batch_size // self.micro_batch_size


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration for SFT on instruction/chat data.

    Mix of tool-calling/coding (60%) and conversation/reasoning (40%).
    Each entry: (dataset_name, split, n_samples, category).
    n_samples=0 means use full dataset.
    """

    # Dataset mixture — loaded and merged by ChatSFTDataset
    # All non-gated, publicly available on HuggingFace.
    datasets: Tuple[Tuple[str, str, int, str], ...] = (
        # Instruction following + coding (~60%)
        ("tatsu-lab/alpaca", "train", 15_000, "instruction"),
        ("databricks/databricks-dolly-15k", "train", 10_000, "instruction"),
        # Conversation / reasoning (~40%)
        ("HuggingFaceH4/no_robots", "train", 5_000, "conversation"),
    )

    # Sequence configuration
    max_seq_len: int = 1024

    # Streaming shuffle buffer
    buffer_size: int = 10_000

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True

    # Train/eval split
    test_split_ratio: float = 0.05             # 5% held out for eval (~1500 samples)

    # Thinking mode configuration
    enable_thinking: bool = False               # Default: non-thinking mode
    thinking_system_prompt: str = (
        "You are a helpful assistant. Before responding, think through your reasoning "
        "step by step inside <thinkong>...</thinking> tags, then provide your final answer."
    )
    fast_system_prompt: str = "You are a helpful assistant."
