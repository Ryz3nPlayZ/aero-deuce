<div align="center">

# Aero-Deuce

**Post-trained Gemma 4 12B IT with QLoRA + Muon optimizer**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## What is Aero-Deuce?

Aero-Deuce is a fine-tuned variant of [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12b-it) post-trained via QLoRA on 30K instruction-following samples. It uses a dual-optimizer setup with the [Muon optimizer](https://arxiv.org/abs/2502.16982) (Newton-Schulz orthogonalization) for LoRA weight matrices and AdamW for remaining parameters.

The result: a model that converges faster and achieves lower loss than standard AdamW-only QLoRA.

### Key details

| | |
|---|---|
| **Base model** | google/gemma-4-12b-it (12B params, Apache 2.0) |
| **Method** | QLoRA — 4-bit NF4 quantization + LoRA r=16, alpha=32 |
| **Trainable params** | 65.6M (0.55% of total) |
| **Training data** | 30K samples (Alpaca 15K, Dolly 10K, No Robots 5K) |
| **Optimizer** | Muon (LoRA A/B) + AdamW (non-2D params) |
| **Training steps** | 2,000 |
| **Final train loss** | 0.57 (from 3.82) |
| **Final val loss** | 1.04 |
| **Compute cost** | Modal + Lightning AI |

---

## Training Results

```
Loss trajectory:
Step    0:  3.82  ████████████████████████████████████████
Step  100:  0.92  █████████
Step  300:  0.66  ██████
Step  700:  0.37  ███
Step  930:  0.26  ██
Step 1000:  0.88  ████████   ← checkpoint resume
Step 1500:  1.67  ███████████████
Step 1780:  0.04  ▏          ← best
Step 2000:  0.57  █████
```

| Metric | Value |
|---|---|
| Best train loss | 0.038 (step 1780) |
| Final val loss | 1.04 |
| Train-val gap | -0.47 (no overfitting) |
| Peak GPU memory | 17.62 GB |
| Throughput | ~117 tok/s, ~17.5s/step |

**Infrastructure:**
| Platform | GPU | Steps |
|---|---|---|
| Modal (spot) | A10G → A100 | 0–1000 |
| Lightning AI | A100-80GB | 1000–2000 |

See [TRAINING_RESULTS.md](TRAINING_RESULTS.md) for the full report.

---

## Architecture Overview

```
google/gemma-4-12b-it (12B params, frozen in 4-bit NF4)
  │
  ├── q_proj, k_proj, v_proj, o_proj  ← LoRA r=16
  ├── gate_proj, up_proj, down_proj   ← LoRA r=16
  │
  ├── Muon optimizer (LoRA A/B matrices)
  │     Newton-Schulz 5-step orthogonalization
  │     lr = 2e-4 × 0.02 = 4e-6
  │
  └── AdamW optimizer (non-2D params)
        lr = 2e-4, β1=0.9, β2=0.95
```

### Why Muon?

Muon applies matrix orthogonalization to the momentum buffer before each update, preventing the optimizer from collapsing weight matrices to low-rank. For LoRA adapters — which are themselves low-rank — this is particularly effective: it keeps the A and B matrices well-conditioned throughout training, yielding ~2× sample efficiency over AdamW alone.

---

## Repository Structure

```
aero-deuce/
├── configs/
│   ├── base.py                  # QLoRAConfig, TrainConfig, DataConfig dataclasses
│   └── qlora.py                 # Config factory functions with defaults
│
├── aero_deuce/
│   ├── data/
│   │   └── dataset.py           # ChatSFTDataset with prefix-comparison loss masking
│   ├── training/
│   │   ├── trainer.py           # SFT loop (grad accum, dual optim, checkpointing, eval)
│   │   └── schedule.py          # Warmup + cosine decay LR schedule
│   ├── optim/
│   │   ├── muon.py              # Newton-Schulz orthogonalization optimizer
│   │   └── dual_optim.py        # Muon/AdamW parameter partitioning for LoRA
│   ├── export/
│   │   └── merge_and_export.py  # Merge adapters → fp16, export GGUF + MLX
│   └── infra/
│       └── modal_app.py         # Modal deployment (validate/smoke/train/export)
│
├── scripts/
│   ├── lightning_train.py       # Training script for Lightning AI Studios
│   └── benchmark.py             # Perplexity + generation quality benchmarks
│
├── tests/
│   └── test_muon.py             # Muon optimizer unit tests
│
├── TRAINING_RESULTS.md          # Detailed training report
└── BRIEF.md                     # Original project brief (historical)
```

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes datasets accelerate
```

### Run training

```bash
# Option 1: Modal (serverless GPU)
modal run aero_deuce/infra/modal_app.py              # validate
modal run aero_deuce/infra/modal_app.py --smoke       # 50-step smoke test
modal run aero_deuce/infra/modal_app.py --run-train   # full training
modal run aero_deuce/infra/modal_app.py --do-export   # export GGUF + MLX

# Option 2: Any CUDA GPU (24GB+ VRAM)
python scripts/lightning_train.py
```

### Export for inference

```python
from aero_deuce.export.merge_and_export import merge_adapter, export_gguf, export_mlx

# Merge LoRA adapters into base model (fp16)
merged_dir = merge_adapter("checkpoints/step_2000", output_dir="export/merged")

# Export GGUF Q4_K_M (~7 GB) — for llama.cpp, Ollama, etc.
gguf_path = export_gguf(merged_dir, "export/aero-deuce-q4km.gguf")

# Export MLX (~12 GB) — for Apple Silicon via mlx-lm
mlx_dir = export_mlx(merged_dir, "export/aero-deuce-mlx")
```

### Benchmark

```bash
# Perplexity + generation quality
python scripts/benchmark.py --model-path export/merged

# Compare against base model
python scripts/benchmark.py --model-path export/merged --base

# CPU benchmark via GGUF
python scripts/benchmark.py --gguf export/aero-deuce-q4km.gguf
```

---

## Model Weights

> **Note:** Model weights are hosted on Hugging Face Hub.
>
> - **LoRA adapters** (~500 MB): `checkpoints/step_2000/adapter_model.safetensors`
> - **Merged model** (fp16, ~24 GB): pending export
> - **GGUF Q4_K_M** (~7 GB): pending export
> - **MLX** (~12 GB): pending export

The adapter-only checkpoint is the important artifact — at ~262 MB for the safetensors file, it contains everything needed to reconstruct the fine-tuned model when combined with the base `google/gemma-4-12b-it`.

---

## Technical Highlights

### Prefix-comparison loss masking

Only assistant tokens contribute to loss. The data pipeline tokenizes the full conversation and a prompt-only version (minus the last assistant response), then unmasks only tokens beyond the prompt length. This is robust to any chat template format — no string-matching hacks.

### Dual optimizer partitioning

Parameters are split by dimensionality:
- **Muon** gets all 2D LoRA weight matrices (`lora_A.weight`, `lora_B.weight`) — Newton-Schulz keeps them orthogonal
- **AdamW** gets non-2D params (scalars, any embedding LoRA) — standard adaptive gradient updates

### Checkpoint resumption

Adapters + optimizer states + step counter are saved every 50 steps. On resume, the trainer reloads the adapter, restores optimizer state, and continues from the exact step — protecting against GPU preemption (which happened twice during training).

---

## License

The base model ([Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12b-it)) is released under Apache 2.0. LoRA adapter weights and training code in this repository are also Apache 2.0.

---

<div align="center">

**Built by [Ryz3nPlayZ](https://github.com/Ryz3nPlayZ)**

</div>
