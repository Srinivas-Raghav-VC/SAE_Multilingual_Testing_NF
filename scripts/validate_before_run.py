#!/usr/bin/env python3
"""Pre-flight Validation Script for A100 GPU Runs.

Run this before executing experiments on the A100 VM to ensure everything
is properly configured.

Usage:
    python scripts/validate_before_run.py [--quick] [--use-9b]

Checks:
    1. GPU availability and memory
    2. Required environment variables
    3. Python dependencies
    4. Model/SAE download verification
    5. Data loading test
    6. Quick sanity test (optional)
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_gpu():
    """Check GPU availability and memory."""
    print("\n1. GPU CHECK")
    print("-" * 40)

    try:
        import torch

        if not torch.cuda.is_available():
            print("  [FAIL] CUDA not available!")
            return False

        n_gpus = torch.cuda.device_count()
        print(f"  [OK] CUDA available, {n_gpus} GPU(s) detected")

        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}")
            print(f"         Total memory: {total_mem:.1f} GB")

            if total_mem < 35:
                print(f"  [WARN] GPU {i} has less than 35GB - 9B model may not fit!")

            # Check current memory
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"         Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        return True

    except Exception as e:
        print(f"  [FAIL] GPU check failed: {e}")
        return False


def check_env_vars():
    """Check required environment variables."""
    print("\n2. ENVIRONMENT VARIABLES")
    print("-" * 40)

    required = {
        "HF_TOKEN": "HuggingFace authentication (for model downloads)",
    }

    optional = {
        "GOOGLE_API_KEY": "Gemini API (for LLM-as-judge in exp11, exp22)",
        "GEMINI_API_KEY": "Alternative name for Gemini API",
        "USE_9B": "Enable 9B model variant",
        "SAE_CACHE_SIZE": "Limit SAE cache to save VRAM",
    }

    all_ok = True

    for var, desc in required.items():
        val = os.environ.get(var)
        if val:
            print(f"  [OK] {var}: set ({len(val)} chars)")
        else:
            print(f"  [WARN] {var}: not set - {desc}")
            # HF_TOKEN is only required for gated models

    for var, desc in optional.items():
        val = os.environ.get(var)
        if val:
            print(f"  [OK] {var}: {val if len(val) < 20 else val[:10] + '...'}")
        else:
            print(f"  [INFO] {var}: not set (optional - {desc})")

    return all_ok


def check_dependencies():
    """Check Python dependencies."""
    print("\n3. PYTHON DEPENDENCIES")
    print("-" * 40)

    required = [
        ("torch", "pytorch"),
        ("transformers", "huggingface transformers"),
        ("sae_lens", "SAE library"),
        ("datasets", "huggingface datasets"),
        ("scipy", "scientific computing"),
        ("sklearn", "machine learning"),
        ("tqdm", "progress bars"),
    ]

    optional = [
        ("google.generativeai", "Gemini API"),
        ("matplotlib", "plotting"),
        ("seaborn", "statistical plots"),
    ]

    all_ok = True

    for pkg, desc in required:
        try:
            __import__(pkg.split(".")[0])
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg}: not installed - {desc}")
            all_ok = False

    for pkg, desc in optional:
        try:
            __import__(pkg.split(".")[0])
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [INFO] {pkg}: not installed (optional - {desc})")

    return all_ok


def check_config():
    """Check configuration validity."""
    print("\n4. CONFIGURATION")
    print("-" * 40)

    try:
        from config import (
            MODEL_ID,
            SAE_RELEASE,
            SAE_WIDTH,
            TARGET_LAYERS,
            N_SAMPLES_DISCOVERY,
            SEED,
            USE_9B,
        )

        print(f"  Model: {MODEL_ID}")
        print(f"  SAE: {SAE_RELEASE} ({SAE_WIDTH})")
        print(f"  Layers: {TARGET_LAYERS}")
        print(f"  9B mode: {'ENABLED' if USE_9B else 'disabled'}")
        print(f"  Sample size: {N_SAMPLES_DISCOVERY}")
        print(f"  Seed: {SEED}")

        # Validate layers
        if "9b" in MODEL_ID.lower():
            max_layers = 42
        else:
            max_layers = 26

        for layer in TARGET_LAYERS:
            if layer >= max_layers:
                print(f"  [WARN] Layer {layer} may be out of range for {MODEL_ID}")

        print("  [OK] Configuration valid")
        return True

    except Exception as e:
        print(f"  [FAIL] Configuration error: {e}")
        return False


def check_data():
    """Check data loading."""
    print("\n5. DATA LOADING")
    print("-" * 40)

    try:
        from data import load_research_data

        print("  Loading FLORES data (small sample)...")
        data = load_research_data(
            max_train_samples=10,
            max_test_samples=0,
            max_eval_samples=0,
            seed=42,
        )

        n_langs = len(data.train)
        print(f"  [OK] Loaded {n_langs} languages")
        for lang, texts in list(data.train.items())[:3]:
            print(f"       {lang}: {len(texts)} texts")

        return True

    except Exception as e:
        print(f"  [FAIL] Data loading failed: {e}")
        return False


def quick_sanity_test():
    """Run a quick model sanity test."""
    print("\n6. QUICK SANITY TEST")
    print("-" * 40)

    try:
        import torch
        from model import GemmaWithSAE

        print("  Loading model...")
        model = GemmaWithSAE()
        model.load_model()

        # Test generation
        print("  Testing generation...")
        output = model.generate("Hello, how are", max_new_tokens=10)
        print(f"  Output: '{output[:50]}...'")

        # Test SAE
        print("  Testing SAE activation...")
        layer = 10 if "2b" in model.model_id.lower() else 20
        acts = model.get_sae_activations("Hello world", layer)
        print(f"  SAE shape: {acts.shape}")
        print(f"  Active features: {(acts > 0).sum().item()}")

        # Memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"  GPU memory used: {allocated:.2f} GB")

        print("  [OK] Sanity test passed")
        return True

    except Exception as e:
        print(f"  [FAIL] Sanity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_recommendations():
    """Print recommendations for A100 runs."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR A100 RUNS")
    print("=" * 60)

    recommendations = [
        "Set USE_9B=1 to use the larger 9B model",
        "Set SAE_CACHE_SIZE=3 to limit SAE memory usage",
        "Run with: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "Monitor GPU memory with: watch -n 1 nvidia-smi",
        "Consider running experiments in batches to avoid OOM",
        "Use --use_9b flag or set USE_9B=1 environment variable",
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-flight validation for A100 runs")
    parser.add_argument("--quick", action="store_true", help="Skip quick sanity test")
    parser.add_argument("--use-9b", action="store_true", help="Set USE_9B=1 for test")
    args = parser.parse_args()

    if args.use_9b:
        os.environ["USE_9B"] = "1"

    print("=" * 60)
    print("SAE MULTILINGUAL STEERING - PRE-FLIGHT VALIDATION")
    print("=" * 60)

    results = []

    results.append(("GPU", check_gpu()))
    results.append(("ENV", check_env_vars()))
    results.append(("DEPS", check_dependencies()))
    results.append(("CONFIG", check_config()))
    results.append(("DATA", check_data()))

    if not args.quick:
        results.append(("SANITY", quick_sanity_test()))
    else:
        print("\n6. QUICK SANITY TEST")
        print("-" * 40)
        print("  [SKIP] Use --no-quick to run")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = True
    for name, ok in results:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All checks passed! Ready to run experiments.")
        print_recommendations()
    else:
        print("\n  Some checks failed. Please fix issues before running experiments.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
