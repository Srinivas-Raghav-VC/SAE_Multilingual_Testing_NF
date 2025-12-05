#!/usr/bin/env python3
"""
Unified experiment runner for SAE Multilingual Steering Research.

This script properly calls the main() functions from each experiment.

Usage:
    python run.py --exp1          # Feature discovery (H1, H3)
    python run.py --exp2          # Steering comparison (H2)
    python run.py --exp3          # Hindi-Urdu overlap (H4)
    python run.py --all           # Run all experiments
    python run.py --validate      # Validate setup only
"""

import argparse
import sys
import os
from pathlib import Path


def validate_setup():
    """Validate all dependencies and configuration."""
    print("=" * 60)
    print("VALIDATING SETUP")
    print("=" * 60)
    
    checks = []
    
    # 1. Check imports
    print("\n1. Checking imports...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
        checks.append(True)
    except ImportError:
        print("   ✗ PyTorch not installed")
        checks.append(False)
    
    try:
        import transformers
        print(f"   ✓ Transformers {transformers.__version__}")
        checks.append(True)
    except ImportError:
        print("   ✗ Transformers not installed")
        checks.append(False)
    
    try:
        from sae_lens import SAE
        print("   ✓ SAE-lens available")
        checks.append(True)
    except ImportError:
        print("   ✗ SAE-lens not installed (pip install sae-lens)")
        checks.append(False)
    
    try:
        import datasets
        print("   ✓ Datasets available")
        checks.append(True)
    except ImportError:
        print("   ✗ Datasets not installed")
        checks.append(False)
    
    # 2. Check config
    print("\n2. Checking config...")
    try:
        from config import (
            MODEL_NAME, MODEL_ID, SAE_RELEASE, HIDDEN_DIM,
            N_SAMPLES_DISCOVERY, N_SAMPLES_EVAL, NUM_FEATURES_OPTIONS,
            TARGET_LAYERS, MONOLINGUALITY_THRESHOLD, STEERING_STRENGTHS,
            LANGUAGES, ATTN_IMPLEMENTATION
        )
        print(f"   ✓ Config loaded successfully")
        print(f"     Model: {MODEL_NAME}")
        print(f"     SAE: {SAE_RELEASE}")
        print(f"     Layers: {TARGET_LAYERS}")
        print(f"     N_SAMPLES_DISCOVERY: {N_SAMPLES_DISCOVERY}")
        print(f"     N_SAMPLES_EVAL: {N_SAMPLES_EVAL}")
        checks.append(True)
    except ImportError as e:
        print(f"   ✗ Config error: {e}")
        checks.append(False)
    
    # 3. Check data module
    print("\n3. Checking data module...")
    try:
        from data import load_flores
        print("   ✓ Data module loads")
        checks.append(True)
    except ImportError as e:
        print(f"   ✗ Data module error: {e}")
        checks.append(False)
    
    # 4. Check model module
    print("\n4. Checking model module...")
    try:
        from model import GemmaWithSAE
        print("   ✓ Model module loads")
        checks.append(True)
    except ImportError as e:
        print(f"   ✗ Model module error: {e}")
        checks.append(False)
    
    # 5. Check experiment modules
    print("\n5. Checking experiment modules...")
    for exp_name in ["exp1_feature_discovery", "exp2_steering", "exp3_hindi_urdu"]:
        try:
            module = __import__(f"experiments.{exp_name}", fromlist=["main"])
            if hasattr(module, "main"):
                print(f"   ✓ {exp_name} has main()")
                checks.append(True)
            else:
                print(f"   ✗ {exp_name} missing main()")
                checks.append(False)
        except ImportError as e:
            print(f"   ✗ {exp_name} import error: {e}")
            checks.append(False)
    
    # 6. Check GPU
    print("\n6. Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            checks.append(True)
        else:
            print("   ⚠ No CUDA GPU (will be slow)")
            checks.append(True)  # Not fatal
    except Exception as e:
        print(f"   ⚠ GPU check failed: {e}")
        checks.append(True)  # Not fatal
    
    # 7. Check HF token
    print("\n7. Checking HuggingFace token...")
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"   ✓ HF_TOKEN set ({len(hf_token)} chars)")
        checks.append(True)
    else:
        print("   ⚠ HF_TOKEN not set (may need for gated models)")
        checks.append(True)  # Not fatal for public models
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"VALIDATION: {passed}/{total} checks passed")
    print("=" * 60)
    
    return all(checks)


def run_exp1():
    """Run Experiment 1: Feature Discovery (H1, H3)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Feature Discovery")
    print("=" * 60)
    print("H1: SAEs contain ≥10 robust Hindi-specific features per layer")
    print("H3: Mid-layers (40-60% depth) contain most language features")
    print("=" * 60 + "\n")
    
    from experiments.exp1_feature_discovery import main
    main()


def run_exp2():
    """Run Experiment 2: Steering Comparison (H2)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Steering Method Comparison")
    print("=" * 60)
    print("H2: Activation-diff features outperform monolinguality by ≥5%")
    print("Methods: activation_diff, monolinguality, random, dense")
    print("=" * 60 + "\n")
    
    from experiments.exp2_steering import main
    main()


def run_exp3():
    """Run Experiment 3: Hindi-Urdu Overlap (H4)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Hindi-Urdu Feature Overlap")
    print("=" * 60)
    print("H4: Hindi-Urdu share >80% features (same language, different script)")
    print("Testing semantic vs script feature separation")
    print("=" * 60 + "\n")
    
    from experiments.exp3_hindi_urdu import main
    main()


def run_all():
    """Run all experiments sequentially."""
    print("\n" + "=" * 60)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 60)
    
    experiments = [
        ("exp1", run_exp1),
        ("exp2", run_exp2),
        ("exp3", run_exp3),
    ]
    
    results = {}
    for name, func in experiments:
        print(f"\n{'='*60}")
        print(f"Starting {name}...")
        print("=" * 60)
        try:
            func()
            results[name] = "SUCCESS"
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            results[name] = f"FAILED: {e}"
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {symbol} {name}: {status}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="SAE Multilingual Steering Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --validate       # Check setup
  python run.py --exp1           # Feature discovery
  python run.py --exp2           # Steering comparison  
  python run.py --exp3           # Hindi-Urdu overlap
  python run.py --all            # Run everything
        """
    )
    
    parser.add_argument("--validate", action="store_true", help="Validate setup only")
    parser.add_argument("--exp1", action="store_true", help="Run exp1: Feature Discovery")
    parser.add_argument("--exp2", action="store_true", help="Run exp2: Steering Comparison")
    parser.add_argument("--exp3", action="store_true", help="Run exp3: Hindi-Urdu Overlap")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    
    args = parser.parse_args()
    
    # Default: show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START")
        print("=" * 60)
        print("1. Validate setup:  python run.py --validate")
        print("2. Run all:         python run.py --all")
        print("=" * 60)
        return
    
    # Validate first if requested
    if args.validate:
        success = validate_setup()
        sys.exit(0 if success else 1)
    
    # Run specific experiments
    if args.exp1:
        run_exp1()
    
    if args.exp2:
        run_exp2()
    
    if args.exp3:
        run_exp3()
    
    if args.all:
        run_all()


if __name__ == "__main__":
    main()
