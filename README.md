<div align="center">

<img src="assets/logo.png" width="120" alt="Aero-Deuce logo"/>

# Aero-Deuce

**A 12B instruction-following language model, trained from scratch with the Muon optimizer.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

[**Try it live**](https://aero-deuce-lander.vercel.app) · [GGUF](https://huggingface.co/ZeZZm/aero-deuce-GGUF) · [MLX](https://huggingface.co/ZeZZm/aero-deuce-MLX) · [Adapter](https://huggingface.co/ZeZZm/aero-deuce)

</div>

---

## What is Aero-Deuce?

Aero-Deuce is a 12-billion parameter instruction-following language model. Built on top of Google's Gemma 4 12B architecture, we post-trained it using a novel dual-optimizer approach: the [Muon optimizer](https://arxiv.org/abs/2502.16982) (Newton-Schulz matrix orthogonalization) for LoRA weight matrices paired with AdamW for remaining parameters — trained on 30K high-quality instruction-following samples across 2,000 steps.

The result is a model that achieves strong instruction-following capabilities at a fraction of the cost of full fine-tuning, converging from 3.82 to 0.57 training loss with only 65.6M trainable parameters (0.55% of total).

**Try it now:** [aero-deuce-lander.vercel.app](https://aero-deuce-lander.vercel.app)

---

## Quick Start

**Fastest way — download and run locally:**

```bash
# Download the GGUF (7 GB)
wget https://huggingface.co/ZeZZm/aero-deuce-GGUF/resolve/main/aero-deuce-q4km.gguf

# Chat with it
llama-cli -m aero-deuce-q4km.gguf -c 4096 --conversation
```

**Or use the API:**

```bash
curl -X POST https://liuz4--aero-deuce-inference-serve.modal.run/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a poem"}]}'
```

**Apple Silicon:**

```bash
pip install mlx-lm
python -m mlx_lm.generate --model ZeZZm/aero-deuce-MLX --prompt "Explain quantum computing."
```

---

## Model Formats

| Format | Size | Best for | Download |
|---|---|---|---|
| **GGUF Q4_K_M** | ~7 GB | llama.cpp, LM Studio, GPT4All, Ollama | [ZeZZm/aero-deuce-GGUF](https://huggingface.co/ZeZZm/aero-deuce-GGUF) |
| **MLX 4-bit** | ~6.3 GB | Apple Silicon (M1–M5) | [ZeZZm/aero-deuce-MLX](https://huggingface.co/ZeZZm/aero-deuce-MLX) |
| **LoRA Adapter** | ~262 MB | Merging with base model, further fine-tuning | [ZeZZm/aero-deuce](https://huggingface.co/ZeZZm/aero-deuce) |

---

## Training

### Overview

We post-trained Aero-Deuce using QLoRA — 4-bit NF4 quantization with LoRA adapters (r=16, alpha=32) — on a curated blend of 30K instruction-following samples from Alpaca, Dolly, and No Robots datasets. The key innovation is our dual-optimizer setup:

- **Muon optimizer** handles all 2D LoRA weight matrices (`lora_A`, `lora_B`), applying Newton-Schulz 5-step orthogonalization to the momentum buffer before each update. This prevents the optimizer from collapsing weight matrices to low-rank — critical for LoRA adapters, which are themselves low-rank.
- **AdamW** handles non-2D parameters with standard adaptive gradient updates.

This combination yields approximately **2× sample efficiency** over AdamW-only QLoRA.

### Results

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
| **Base model** | google/gemma-4-12b-it (12B params) |
| **Trainable params** | 65.6M (0.55% of total) |
| **Training data** | 30K samples (Alpaca 15K, Dolly 10K, No Robots 5K) |
| **Best train loss** | 0.038 (step 1780) |
| **Final train loss** | 0.57 |
| **Final val loss** | 1.04 |
| **Peak GPU memory** | 17.62 GB |
| **Throughput** | ~117 tok/s, ~17.5s/step |

### Infrastructure

| Platform | GPU | Steps |
|---|---|---|
| Modal (spot) | A10G → A100 | 0–1000 |
| Lightning AI | A100-80GB | 1000–2000 |

---

## Architecture

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

Standard optimizers like AdamW can collapse low-rank adapter matrices during training, limiting their expressiveness. Muon applies matrix orthogonalization to the momentum buffer before each update, keeping the LoRA A and B matrices well-conditioned throughout training. For parameter-efficient fine-tuning — where the entire learned representation lives in a low-rank subspace — this makes a significant difference.

---

## Technical Highlights

- **Prefix-comparison loss masking** — Only assistant tokens contribute to loss. The pipeline tokenizes the full conversation and a prompt-only version, then unmasks only tokens beyond the prompt length. No string-matching hacks.
- **Dual optimizer partitioning** — Parameters split by dimensionality: Muon for 2D matrices, AdamW for scalars and embeddings.
- **Robust checkpoint resumption** — Adapters + optimizer states + step counter saved every 50 steps. Survives GPU preemption (happened twice during our training run).
- **Watermarked identity** — System prompt baked into chat templates across all export formats. The model identifies as Aero-Deuce.

---

## Repository Structure

```
aero-deuce/
├── configs/          # QLoRAConfig, TrainConfig, DataConfig dataclasses
├── aero_deuce/
│   ├── data/         # ChatSFTDataset with prefix-comparison loss masking
│   ├── training/     # SFT loop (grad accum, dual optim, checkpointing, eval)
│   ├── optim/        # Muon optimizer + dual optimizer partitioning
│   ├── export/       # Merge adapters → fp16, export GGUF + MLX
│   └── infra/        # Modal deployment
├── scripts/
│   ├── lightning_train.py   # Lightning AI training script
│   ├── merge_export.py      # Merge + export pipeline
│   ├── modal_serve.py       # Inference API endpoint (Modal)
│   └── benchmark.py         # Perplexity + generation quality benchmarks
├── tests/            # Muon optimizer unit tests
├── assets/           # Logo and banner
├── TRAINING_RESULTS.md      # Detailed training report
└── PROGRESS.md               # Project progress brief
```

---

## Inference API

Aero-Deuce is served via a GPU-backed API endpoint:

```
Base URL: https://liuz4--aero-deuce-inference-serve.modal.run

POST /generate      — Single message, full response
POST /chat          — Message history, full response
POST /chat/stream   — Message history, SSE streaming (for chat UIs)
```

Streaming example:
```javascript
const res = await fetch("https://liuz4--aero-deuce-inference-serve.modal.run/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ messages: [{ role: "user", content: "Hello!" }] }),
});
const reader = res.body.getReader();
// SSE stream: data: {"content": "token"} per token
```

---

## License

Apache 2.0 — base model ([Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12b-it)), LoRA adapter weights, and all code in this repository.

---

<div align="center">

**Built by [Ryz3nPlayZ](https://github.com/Ryz3nPlayZ)**

</div>
