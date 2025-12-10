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
    """Validate all dependencies, configuration, and research rigor checks."""
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
    
    # 5. Check experiment modules (smoke-test main() entrypoints)
    print("\n5. Checking experiment modules...")
    for exp_name in ["exp1_feature_discovery", "exp2_steering", "exp3_hindi_urdu_fixed"]:
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
    
    # 8. Research rigor validation
    print("\n8. Research rigor validation...")
    try:
        from validation import validate_experiment_setup
        rigor_passed, rigor_warnings = validate_experiment_setup(verbose=False)
        if rigor_passed:
            print(f"   ✓ Research rigor checks passed ({len(rigor_warnings)} warnings)")
            checks.append(True)
        else:
            errors = [w for w in rigor_warnings if w.severity == "error"]
            print(f"   ✗ Research rigor checks failed: {len(errors)} critical errors")
            for e in errors:
                print(f"     - {e.message}")
            checks.append(False)
    except ImportError as e:
        print(f"   ⚠ Could not run rigor validation: {e}")
        checks.append(True)  # Not fatal, but should be fixed
    
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
    
    from experiments.exp3_hindi_urdu_fixed import main
    main()

def run_exp4():
    """Run Experiment 4: Language Steering Spillover Analysis."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Language Steering Spillover Analysis")
    print("=" * 60)
    print("Goal: Measure how EN→HI steering spills over to other languages")
    print("=" * 60 + "\n")

    from experiments.exp4_spillover import main as exp4_main
    exp4_main()

def run_exp6():
    """Run Experiment 6: Script vs Semantics Controls."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Script vs Semantics Controls")
    print("=" * 60)
    print("Goal: Separate script-only vs semantic/generative Indic features")
    print("=" * 60 + "\n")

    from experiments.exp6_script_semantics_controls import main
    main()


def run_exp7():
    """Run Experiment 7: Causal Probing of SAE Features."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Causal Probing of SAE Features")
    print("=" * 60)
    print("Goal: Estimate causal effect of individual features on Hindi steering")
    print("=" * 60 + "\n")

    from experiments.exp7_causal_feature_probing import main
    main()


def run_exp8():
    """Run Experiment 8: Scaling to 9B and Low-Resource Languages."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 8: Scaling to 9B and Low-Resource Languages")
    print("=" * 60)
    print("Goal: Compare 2B vs 9B and low-resource behavior")
    print("=" * 60 + "\n")

    from experiments.exp8_scaling_9b_low_resource import main
    main()


def run_exp9():
    """Run Experiment 9: Layer-wise Steering Sweep with LLM Judge."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9: Layer-wise Steering Sweep")
    print("=" * 60)
    print("Goal: Find most effective steering layers and methods per language")
    print("=" * 60 + "\n")

    from experiments.exp9_layer_sweep_steering import run_layer_sweep
    run_layer_sweep()


def run_exp11():
    """Run Experiment 11: Calibrated Gemini Judge."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 11: Calibrated LLM-as-Judge Evaluation (Gemini)")
    print("=" * 60)
    print("Goal: Calibrate Gemini judge for Hindi and German steering")
    print("=" * 60 + "\n")
    
    from experiments.exp11_judge_calibration import main as exp11_main
    exp11_main()


def run_exp12():
    """Run Experiment 12: QA Degradation Under Steering."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 12: QA Degradation Under Steering")
    print("=" * 60)
    print("Goal: Measure how EN→target steering affects MLQA/IndicQA performance")
    print("=" * 60 + "\n")
    
    from experiments.exp12_qa_degradation import main as exp12_main
    exp12_main()


def run_exp13():
    """Run Experiment 13: Script vs Semantic Group Ablation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 13: Group Ablation for Script vs Semantic Features")
    print("=" * 60)
    print("Goal: Test causal impact of script-sensitive vs script-invariant feature groups")
    print("=" * 60 + "\n")

    from experiments.exp13_script_semantic_ablation import main as exp13_main
    exp13_main()


def run_exp14():
    """Run Experiment 14: Cross-Lingual Alignment (Language-Agnostic Space)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 14: Cross-Lingual Alignment (Language-Agnostic Space)")
    print("=" * 60)
    print("Goal: Measure EN+Indic alignment across layers (language-agnostic region)")
    print("=" * 60 + "\n")

    from experiments.exp14_language_agnostic_space import main as exp14_main
    exp14_main()


def run_exp15():
    """Run Experiment 15: Directional Symmetry of Steering."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 15: Directional Symmetry of Steering (EN↔Indic)")
    print("=" * 60)
    print("Goal: Compare EN→Indic vs Indic→EN steering effectiveness")
    print("=" * 60 + "\n")

    from experiments.exp15_directional_symmetry import main as exp15_main
    exp15_main()


def run_exp16():
    """Run Experiment 16: Code-mix and Noise Robustness."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 16: Code-mix and Noise Robustness (EN→HI)")
    print("=" * 60)
    print("Goal: Test steering robustness on code-mixed and noisy prompts")
    print("=" * 60 + "\n")

    from experiments.exp16_code_mix_robustness import main as exp16_main
    exp16_main()


def run_all():
    """Run all core experiments sequentially, then generate plots."""
    print("\n" + "=" * 60)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 60)
    
    experiments = [
        ("exp1_feature_discovery", run_exp1),
        ("exp2_steering_comparison", run_exp2),
        ("exp3_hindi_urdu_overlap", run_exp3),
        ("exp4_spillover", lambda: __import__("experiments.exp4_spillover", fromlist=["main"]).main()),
        ("exp5_hierarchical", lambda: __import__("experiments.exp5_hierarchical", fromlist=["main"]).main()),
        ("exp6_script_semantics_controls", run_exp6),
        ("exp7_causal_feature_probing", run_exp7),
        ("exp8_scaling_9b_low_resource", run_exp8),
        ("exp9_layer_sweep_steering", run_exp9),
        ("exp10_attribution_occlusion", lambda: __import__("experiments.exp10_attribution_occlusion", fromlist=["main"]).main()),
        ("exp11_judge_calibration", run_exp11),
        ("exp12_qa_degradation", run_exp12),
        ("exp13_script_semantic_ablation", run_exp13),
        ("exp14_language_agnostic_space", run_exp14),
        ("exp15_directional_symmetry", run_exp15),
        ("exp16_code_mix_robustness", run_exp16),
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
    
    # Generate plots at the end
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    try:
        import subprocess
        subprocess.run(["python", "plots.py", "--results_dir", "results"], check=False)
        print("✓ Plot generation invoked (see results/figures)")
    except Exception as e:
        print(f"✗ Plot generation failed: {e}")

    # Generate text summary report
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    try:
        import subprocess
        subprocess.run(["python", "summarize_results.py"], check=False)
        print("✓ Summary report generated (see results/summary_report.txt)")
    except Exception as e:
        print(f"✗ Summary report generation failed: {e}")
    
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
    parser.add_argument("--exp4", action="store_true", help="Run exp4: Language Steering Spillover")
    parser.add_argument("--exp6", action="store_true", help="Run exp6: Script vs Semantics Controls")
    parser.add_argument("--exp7", action="store_true", help="Run exp7: Causal Feature Probing")
    parser.add_argument("--exp8", action="store_true", help="Run exp8: 2B vs 9B + Low-Resource Scaling")
    parser.add_argument("--exp9", action="store_true", help="Run exp9: Layer-wise Steering Sweep")
    parser.add_argument("--exp10", action="store_true", help="Run exp10: Occlusion-Based Attribution Steering")
    parser.add_argument("--exp11", action="store_true", help="Run exp11: Calibrated LLM-as-Judge Evaluation")
    parser.add_argument("--exp12", action="store_true", help="Run exp12: QA Degradation Under Steering")
    parser.add_argument("--exp13", action="store_true", help="Run exp13: Group Ablation for Script vs Semantic Features")
    parser.add_argument("--exp14", action="store_true", help="Run exp14: Cross-Lingual Alignment (Language-Agnostic Space)")
    parser.add_argument("--exp15", action="store_true", help="Run exp15: Directional Symmetry of Steering")
    parser.add_argument("--exp16", action="store_true", help="Run exp16: Code-mix and Noise Robustness")
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

    if args.exp4:
        run_exp4()

    if args.exp6:
        run_exp6()

    if args.exp7:
        run_exp7()

    if args.exp8:
        run_exp8()

    if args.exp9:
        run_exp9()

    if args.exp10:
        print("\n" + "=" * 60)
        print("EXPERIMENT 10: Occlusion-Based Attribution Steering")
        print("=" * 60)
        print("Goal: Attribution-selected steering via feature occlusion")
        print("=" * 60 + "\n")
        from experiments.exp10_attribution_occlusion import main as exp10_main
        exp10_main()
    
    if args.exp11:
        run_exp11()
    
    if args.exp12:
        run_exp12()
    
    if args.exp13:
        run_exp13()
    
    if args.exp14:
        run_exp14()

    if args.exp15:
        run_exp15()

    if args.exp16:
        run_exp16()
    
    if args.all:
        run_all()


if __name__ == "__main__":
    main()
