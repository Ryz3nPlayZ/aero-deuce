"""Benchmark Aero-Deuce (Gemma 4 12B IT + QLoRA) against the base model.

Compares the fine-tuned model against the base Gemma 4 12B IT on:
1. Perplexity on a held-out instruction dataset
2. Qualitative response quality (side-by-side generation)
3. Response length and coherence metrics

Usage:
    # After training, benchmark the merged model
    python scripts/benchmark.py --model-path /checkpoints/merged

    # Compare against base model
    python scripts/benchmark.py --base --model-path /checkpoints/merged

    # Quick local test (no GPU needed, uses GGUF)
    python scripts/benchmark.py --gguf /checkpoints/export/aero-deuce-q4km.gguf
"""

import argparse
import json
import time
from pathlib import Path


def benchmark_perplexity(model_path: str, tokenizer_path: str, n_samples: int = 100):
    """Measure perplexity on held-out instruction data.

    Lower perplexity = model predicts the text better.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"[Bench] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Load eval data
    print("[Bench] Loading eval dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(n_samples))

    total_loss = 0.0
    total_tokens = 0

    print(f"[Bench] Computing perplexity on {n_samples} samples...")
    with torch.no_grad():
        for i, example in enumerate(ds):
            # Format as chat
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = enc["input_ids"].to(model.device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

            if (i + 1) % 20 == 0:
                avg_loss = total_loss / total_tokens
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                print(f"  [{i+1}/{n_samples}] PPL: {ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"\n[Bench] Final Perplexity: {perplexity:.2f}")
    return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}


def benchmark_generation(model_path: str, tokenizer_path: str):
    """Qualitative generation comparison on diverse prompts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Bench] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    test_prompts = [
        "Write a Python function that finds the longest palindrome in a string.",
        "Explain quantum computing in simple terms.",
        "What are the key differences between REST and GraphQL APIs?",
        "Write a haiku about programming.",
        "How would you design a URL shortener service? Describe the architecture.",
        "Debug this code: def add(a, b) return a - b",
        "What is the time complexity of binary search and why?",
        "Write a SQL query to find the top 5 customers by total purchases.",
    ]

    results = []
    print(f"\n[Bench] Generating responses for {len(test_prompts)} prompts...\n")

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        elapsed = time.time() - start

        # Decode only the generated tokens
        generated = outputs[0][inputs.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        tokens_generated = len(generated)
        tokens_per_sec = tokens_generated / elapsed

        print(f"Q: {prompt[:60]}...")
        print(f"A: {response[:200]}...")
        print(f"  ({tokens_generated} tokens, {tokens_per_sec:.1f} tok/s, {elapsed:.1f}s)")
        print()

        results.append({
            "prompt": prompt,
            "response": response,
            "tokens": tokens_generated,
            "tok_per_sec": tokens_per_sec,
            "latency_s": elapsed,
        })

    avg_speed = sum(r["tok_per_sec"] for r in results) / len(results)
    print(f"[Bench] Average generation speed: {avg_speed:.1f} tok/s")

    return results


def benchmark_gguf(gguf_path: str):
    """Benchmark GGUF model via llama-cpp-python."""
    from llama_cpp import Llama

    print(f"[Bench] Loading GGUF from {gguf_path}...")
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=0)

    prompts = [
        "Write a Python function to reverse a linked list.",
        "Explain the difference between TCP and UDP.",
        "What is the halting problem?",
    ]

    for prompt in prompts:
        start = time.time()
        response = llm(
            f"[INST] {prompt} [/INST]",
            max_tokens=200,
            temperature=0.7,
        )
        elapsed = time.time() - start
        text = response["choices"][0]["text"]
        print(f"Q: {prompt[:60]}...")
        print(f"A: {text[:200]}...")
        print(f"  ({response['usage']['completion_tokens']} tokens, {elapsed:.1f}s)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Aero-Deuce")
    parser.add_argument("--model-path", type=str, help="Path to merged model")
    parser.add_argument("--base", action="store_true", help="Also benchmark base model for comparison")
    parser.add_argument("--gguf", type=str, help="Path to GGUF file for CPU benchmark")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples for perplexity")
    args = parser.parse_args()

    if args.gguf:
        benchmark_gguf(args.gguf)
        return

    if not args.model_path:
        print("Specify --model-path or --gguf")
        return

    model_path = args.model_path
    tokenizer_path = args.model_path

    # Perplexity benchmark
    print("=" * 60)
    print("  PERPLEXITY BENCHMARK")
    print("=" * 60)
    ft_ppl = benchmark_perplexity(model_path, tokenizer_path, args.n_samples)

    if args.base:
        base_path = "google/gemma-4-12b-it"
        print(f"\n[Bench] Comparing against base model ({base_path})...")
        base_ppl = benchmark_perplexity(base_path, base_path, args.n_samples)
        improvement = base_ppl["perplexity"] - ft_ppl["perplexity"]
        print(f"\n{'='*60}")
        print(f"  Base PPL:     {base_ppl['perplexity']:.2f}")
        print(f"  Fine-tuned:   {ft_ppl['perplexity']:.2f}")
        print(f"  Improvement:  {improvement:+.2f} ({'better' if improvement > 0 else 'worse'})")
        print(f"{'='*60}")

    # Generation benchmark
    print("\n" + "=" * 60)
    print("  GENERATION BENCHMARK")
    print("=" * 60)
    results = benchmark_generation(model_path, tokenizer_path)

    # Save results
    output = {
        "perplexity": ft_ppl,
        "generations": results,
    }
    output_path = Path(model_path) / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Bench] Results saved to {output_path}")


if __name__ == "__main__":
    main()
