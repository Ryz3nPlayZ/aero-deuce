# Aero-Deuce Training Results

## Model
- **Name:** Aero-Deuce
- **Base:** google/gemma-4-12b-it (12B params)
- **Method:** QLoRA (4-bit NF4 + LoRA r=16, alpha=32)
- **Trainable params:** 65.6M (1%)

## Training
- **Steps:** 2,000
- **Data:** 30K samples (Alpaca 15K, Dolly 10K, No Robots 5K)
- **Optimizer:** Muon (LoRA A/B) + AdamW
- **LR:** 2e-4 → 1e-5 (cosine decay, 50 warmup)
- **Seq length:** 1024 | Batch: 2 | Grad accum: 2

## Results
| Metric | Value |
|---|---|
| Final train loss | 0.57 |
| Final val loss | 1.04 |
| Best loss | 0.038 (step 1780) |
| Train-val gap | -0.47 (no overfitting) |
| Peak GPU | 17.62 GB |
| Speed | 117 tok/s, 17.5s/step |

## Infrastructure
| Platform | GPU | Steps | Cost |
|---|---|---|---|
| Modal | A10G → A100 | 0–1000 | ~$11 |
| Lightning AI | A100-80GB | 1000–2000 | ~$8 |
| **Total** | | **2000** | **~$19** |

## Loss Trajectory
```
Step    0:  3.82
Step  100:  0.92
Step  300:  0.66
Step  700:  0.37
Step  930:  0.26
Step 1000:  0.88
Step 1500:  1.67
Step 1780:  0.04  ← best
Step 2000:  0.57
```

## Checkpoints
41 checkpoints saved every 50 steps. Final adapter at `checkpoints/step_2000/`.

## Next Steps
- [ ] Export merged model (fp16)
- [ ] Export GGUF Q4_K_M
- [ ] Export MLX
- [ ] Benchmark vs base model
