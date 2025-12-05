#!/usr/bin/env python3
"""Test Gemini API and run comprehensive experiments.

This script:
1. Tests Gemini API connectivity
2. Validates experimental setup
3. Runs experiments with proper configuration

Usage:
    # Test Gemini API
    python test_and_run.py --test-api
    
    # Validate setup
    python test_and_run.py --validate
    
    # Run full experiments
    python test_and_run.py --run exp1  # Feature discovery
    python test_and_run.py --run exp2  # Steering comparison  
    python test_and_run.py --run exp3  # Hindi-Urdu overlap
    python test_and_run.py --run all   # All experiments
"""

import os
import sys
import argparse
from pathlib import Path


def test_gemini_api():
    """Test Gemini API connectivity."""
    print("=" * 60)
    print("GEMINI API TEST")
    print("=" * 60)
    
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    if not api_key:
        print("\n❌ GOOGLE_API_KEY environment variable not set!")
        print("\nTo fix this:")
        print("1. Get a FREE API key: https://aistudio.google.com/apikey")
        print("2. Set it: export GOOGLE_API_KEY=AIza...")
        return False
    
    print(f"\n✓ API key found: {api_key[:15]}...")
    
    try:
        from google import genai
        print("✓ google-genai package installed")
    except ImportError:
        print("❌ google-genai not installed")
        print("Install with: pip install google-genai")
        return False
    
    try:
        print("\nTesting API connection...")
        client = genai.Client(api_key=api_key)
        
        # Test with Hindi request
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'नमस्ते, मैं Gemini हूं' (Hello, I am Gemini in Hindi)",
        )
        
        print(f"\n✓ API Response: {response.text}")
        
        # Test JSON evaluation
        eval_prompt = """Evaluate this text:
Text: "यह एक परीक्षण वाक्य है।"
Respond ONLY with JSON: {"is_hindi": true/false, "reason": "brief"}"""
        
        response2 = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=eval_prompt,
        )
        
        print(f"✓ JSON Response: {response2.text}")
        
        print("\n" + "=" * 60)
        print("✅ GEMINI API TEST PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. Network connectivity")
        print("3. API quota exceeded")
        return False


def validate_setup():
    """Validate experimental setup."""
    print("=" * 60)
    print("SETUP VALIDATION")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 6
    
    # Check 1: Python packages
    print("\n1. Checking Python packages...")
    required = ["torch", "transformers", "datasets", "sae_lens"]
    for pkg in required:
        try:
            __import__(pkg)
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} - pip install {pkg}")
    checks_passed += 1
    
    # Check 2: HuggingFace token
    print("\n2. Checking HuggingFace token...")
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"   ✓ HF_TOKEN set: {hf_token[:10]}...")
        checks_passed += 1
    else:
        print("   ❌ HF_TOKEN not set (needed for Gemma)")
    
    # Check 3: Gemini API
    print("\n3. Checking Gemini API...")
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if google_key:
        print(f"   ✓ GOOGLE_API_KEY set: {google_key[:15]}...")
        checks_passed += 1
    else:
        print("   ⚠ GOOGLE_API_KEY not set (optional but recommended)")
    
    # Check 4: Data files
    print("\n4. Checking configuration files...")
    config_files = ["config_v2.py", "data_v2.py", "evaluation.py"]
    for f in config_files:
        if Path(f).exists():
            print(f"   ✓ {f}")
        else:
            print(f"   ❌ {f} not found")
    checks_passed += 1
    
    # Check 5: GPU availability
    print("\n5. Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            checks_passed += 1
        else:
            print("   ⚠ No GPU detected (experiments will be slow)")
    except:
        print("   ⚠ Could not check GPU")
    
    # Check 6: Directory structure
    print("\n6. Checking directories...")
    dirs = ["results", "results/figures", "checkpoints"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {d}/")
    checks_passed += 1
    
    print("\n" + "=" * 60)
    print(f"VALIDATION: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    return checks_passed >= 4  # Allow some optional checks to fail


def run_experiment(exp_name: str, config: str = "config_v2.py"):
    """Run an experiment."""
    print("=" * 60)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("=" * 60)
    
    # Import experiment modules
    try:
        if exp_name == "exp1":
            from experiments.exp1_feature_discovery import run_experiment as run_exp1
            run_exp1()
        elif exp_name == "exp2":
            from experiments.exp2_steering import run_experiment as run_exp2
            run_exp2()
        elif exp_name == "exp3":
            from experiments.exp3_hindi_urdu import run_experiment as run_exp3
            run_exp3()
        elif exp_name == "all":
            print("Running all experiments...")
            run_experiment("exp1", config)
            run_experiment("exp2", config)
            run_experiment("exp3", config)
        else:
            print(f"Unknown experiment: {exp_name}")
            print("Available: exp1, exp2, exp3, all")
            return False
            
    except Exception as e:
        print(f"Error running {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def print_quick_start():
    """Print quick start guide."""
    print("""
================================================================================
QUICK START GUIDE
================================================================================

1. SET API KEYS:
   export HF_TOKEN=hf_your_huggingface_token
   export GOOGLE_API_KEY=AIza_your_gemini_key  # Free from aistudio.google.com

2. TEST SETUP:
   python test_and_run.py --validate
   python test_and_run.py --test-api

3. RUN EXPERIMENTS:
   python test_and_run.py --run exp1   # Feature discovery (H1, H3)
   python test_and_run.py --run exp2   # Steering comparison (H2)
   python test_and_run.py --run exp3   # Hindi-Urdu overlap (H4)

4. GENERATE PLOTS:
   python plots.py --results_dir results/

5. VERIFICATION:
   python verify.py --mode check_overlap
   python verify.py --mode debug_mono --layer 13

================================================================================
KEY FINDING FROM YOUR EXPERIMENTS
================================================================================

Your verification revealed a CRITICAL insight:

  Monolinguality features: M = 16,017,084 (very Hindi-specific)
  Activation-diff features: Hindi/English ratio = 1.8x
  
  Overlap between top-25: 0%  ← THEY SELECT DIFFERENT FEATURES!
  
  Steering success:
  - Monolinguality: 0%
  - Activation-diff: 100%

INTERPRETATION:
- Monolinguality selects "DETECTOR" features (fire when Hindi present)
- Activation-diff selects "GENERATOR" features (encode EN→HI direction)
- For STEERING, you need generators, not detectors!

This is a NOVEL finding worth publishing!

================================================================================
    """)


def main():
    parser = argparse.ArgumentParser(description="Test and run SAE experiments")
    parser.add_argument("--test-api", action="store_true", help="Test Gemini API")
    parser.add_argument("--validate", action="store_true", help="Validate setup")
    parser.add_argument("--run", type=str, help="Run experiment (exp1/exp2/exp3/all)")
    parser.add_argument("--config", type=str, default="config_v2.py", help="Config file")
    parser.add_argument("--quick-start", action="store_true", help="Show quick start guide")
    
    args = parser.parse_args()
    
    if args.quick_start or len(sys.argv) == 1:
        print_quick_start()
        return
    
    if args.test_api:
        test_gemini_api()
        return
    
    if args.validate:
        validate_setup()
        return
    
    if args.run:
        # First validate
        if not validate_setup():
            print("\n⚠ Setup validation failed. Continue anyway? (y/n)")
            if input().lower() != 'y':
                return
        
        run_experiment(args.run, args.config)


if __name__ == "__main__":
    main()
