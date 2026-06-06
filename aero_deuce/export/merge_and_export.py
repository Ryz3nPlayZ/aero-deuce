"""Merge LoRA adapters and export to deployment formats.

Post-training pipeline:
1. Load base model in fp16 (not quantized — need full precision for clean merge)
2. Load LoRA adapter checkpoint
3. Merge adapters into base weights via PEFT's merge_and_unload()
4. Export merged model in two formats:
   - GGUF (Q4_K_M): For llama.cpp — cross-platform CPU inference
   - MLX: For Apple Silicon — optimized GPU inference via mlx-lm
"""

from pathlib import Path


def merge_adapter(checkpoint_dir: str, output_dir: str) -> str:
    """Load QLoRA model + adapter and merge in fp16.

    Args:
        checkpoint_dir: Path to LoRA adapter checkpoint (from Trainer.save_checkpoint).
        output_dir: Where to save the merged fp16 model.

    Returns:
        Path to the merged model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model in fp16 (NOT 4-bit — need full precision for merge)
    print(f"[Merge] Loading base model in fp16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-12b-it",
        torch_dtype="auto",
        device_map="auto",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")

    # Load LoRA adapter on top
    print(f"[Merge] Loading LoRA adapter from {checkpoint_dir}...")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    # Merge adapter weights into base model
    print("[Merge] Merging adapter weights into base...")
    merged_model = model.merge_and_unload()

    # Save merged model in fp16
    print(f"[Merge] Saving merged model to {output_dir}...")
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    print(f"[Merge] ✓ Merged model saved to {output_dir}")
    return str(output_dir)


def export_gguf(model_dir: str, output_path: str, quantization: str = "Q4_K_M") -> str:
    """Convert merged fp16 model to GGUF format with quantization.

    Uses llama.cpp's convert script to transform the HuggingFace model
    into GGUF format, then applies quantization.

    Args:
        model_dir: Path to merged fp16 model directory.
        output_path: Path for the output GGUF file.
        quantization: Quantization type (e.g., "Q4_K_M", "Q5_K_M", "Q8_0").

    Returns:
        Path to the quantized GGUF file.
    """
    import subprocess

    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to unquantized GGUF (f16)
    f16_path = output_path.parent / "temp_f16.gguf"
    print(f"[GGUF] Converting {model_dir} to GGUF f16...")

    try:
        # Try using the gguf/llama.cpp conversion tools
        from gguf.convert import convert_to_gguf
        convert_to_gguf(str(model_dir), str(f16_path))
    except ImportError:
        # Fallback: use llama.cpp conversion script
        print("[GGUF] gguf package not available, trying llama-cpp conversion...")
        subprocess.run([
            "python", "-m", "llama_cpp.convert",
            str(model_dir),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ], check=True)

    # Step 2: Quantize to target format
    quantized_path = output_path
    print(f"[GGUF] Quantizing to {quantization}...")

    try:
        from llama_cpp import llama_model_quantize
        llama_model_quantize(str(f16_path), str(quantized_path), quantization)
    except (ImportError, Exception):
        # Fallback: use llama-quantize binary if available
        print("[GGUF] Using llama-quantize binary...")
        subprocess.run([
            "llama-quantize",
            str(f16_path),
            str(quantized_path),
            quantization,
        ], check=True)

    # Cleanup temp f16 file
    if f16_path.exists() and f16_path != quantized_path:
        f16_path.unlink()

    size_mb = quantized_path.stat().st_size / 1e6
    print(f"[GGUF] ✓ {quantized_path} ({size_mb:.0f} MB)")
    return str(quantized_path)


def export_mlx(model_dir: str, output_dir: str) -> str:
    """Convert merged fp16 model to MLX format for Apple Silicon.

    Uses mlx-lm's conversion utility to transform the HuggingFace model
    into Apple MLX format for fast inference on M-series chips.

    Args:
        model_dir: Path to merged fp16 model directory.
        output_dir: Where to save the MLX model.

    Returns:
        Path to the MLX model directory.
    """
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MLX] Converting {model_dir} to MLX format...")

    try:
        # Try using mlx-lm conversion
        from mlx_lm.utils import convert_and_save
        convert_and_save(
            model_name_or_path=str(model_dir),
            output_dir=str(output_dir),
        )
    except ImportError:
        # Fallback: use CLI
        print("[MLX] Using mlx_lm.convert CLI...")
        subprocess.run([
            "python", "-m", "mlx_lm.convert",
            "--hf-path", str(model_dir),
            "--mlx-path", str(output_dir),
        ], check=True)

    print(f"[MLX] ✓ MLX model saved to {output_dir}")
    return str(output_dir)
