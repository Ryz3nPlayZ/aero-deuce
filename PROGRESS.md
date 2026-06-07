we made a custom AI model called Aero-Deuce. it's based on Google's Gemma 4 12B, but we fine-tuned it using a technique called QLoRA with a custom optimizer called Muon. the idea was to make a small, efficient instruction-following model that runs locally on consumer hardware.

we trained it on 30,000 instruction-following examples for 2,000 steps. the training loss dropped from 3.82 to 0.57. it runs at about 10-16 tokens per second on a MacBook Air M5.

the model is exported in three formats: a raw LoRA adapter (for merging with the base model), a GGUF file (runs in llama.cpp, LM Studio, Ollama, etc.), and an MLX version (optimized for Apple Silicon). all three are on HuggingFace. the code is on GitHub.

still need to benchmark it against the base model and submit it to the HuggingFace leaderboard.

code: github.com/Ryz3nPlayZ/aero-deuce
adapter: huggingface.co/ZeZZm/aero-deuce
gguf: huggingface.co/ZeZZm/aero-deuce-GGUF
mlx: huggingface.co/ZeZZm/aero-deuce-MLX
