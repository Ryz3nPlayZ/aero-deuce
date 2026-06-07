---
license: apache-2.0
base_model: google/gemma-4-12b-it
library_name: peft
tags:
- qLoRA
- gemma
- sft
- muon-optimizer
- instruction-following
---

# Aero-Deuce — LoRA Adapter

A fine-tuned Gemma 4 12B instruction-following model trained with QLoRA + Muon optimizer on 30K samples. This repo contains the **LoRA adapter only** (~262 MB) — you need the base model to use it.

## Which format should I use?

| Format | Best for | Link |
|---|---|---|
| **LoRA Adapter** ← you are here | Merging with base model, further fine-tuning | This repo |
| [GGUF Q4_K_M](https://huggingface.co/ZeZZm/aero-deuce-GGUF) | Local inference, llama.cpp, LM Studio, GPT4All (~7 GB) | [ZeZZm/aero-deuce-GGUF](https://huggingface.co/ZeZZm/aero-deuce-GGUF) |
| [MLX 4-bit](https://huggingface.co/ZeZZm/aero-deuce-MLX) | Apple Silicon (Mac), fastest on M-series chips (~6.3 GB) | [ZeZZm/aero-deuce-MLX](https://huggingface.co/ZeZZm/aero-deuce-MLX) |

**Just want to run it locally?** Use the [GGUF version](https://huggingface.co/ZeZZm/aero-deuce-GGUF) — it's one file, no Python needed, works in LM Studio and GPT4All.

## Download

Click **Files and versions** above. The adapter file is `adapter_model.safetensors` (~262 MB).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-12b-it",
    device_map="auto",
    torch_dtype="auto",
)

# Load Aero-Deuce adapter
model = PeftModel.from_pretrained(model, "ZeZZm/aero-deuce")

# Merge for faster inference (optional)
model = model.merge_and_unload()

# Generate
messages = [{"role": "user", "content": "Write a Python function to reverse a linked list."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Model Details

| Property | Value |
|---|---|
| Base Model | google/gemma-4-12b-it (12B params) |
| Training Method | QLoRA — 4-bit NF4 + LoRA r=16, alpha=32 |
| Trainable Params | 65.6M (0.55% of total) |
| Training Data | 30K samples (Alpaca 15K, Dolly 10K, No Robots 5K) |
| Optimizer | Muon (LoRA A/B) + AdamW (non-2D params) |
| Training Steps | 2,000 |
| Final Train Loss | 0.57 (from 3.82) |
| Final Val Loss | 1.04 |
| Adapter Size | ~262 MB |

## Training Infrastructure

| Platform | GPU | Steps |
|---|---|---|
| Modal (spot) | A100 | 0–1000 |
| Lightning AI | A100-80GB | 1000–2000 |

## Source Code

Full training code, configuration, and detailed results: [github.com/Ryz3nPlayZ/aero-deuce](https://github.com/Ryz3nPlayZ/aero-deuce)

## License

Apache 2.0
