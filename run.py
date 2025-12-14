#!/usr/bin/env python3
"""
Unified experiment runner for SAE Multilingual Steering Research.

This script properly calls the main() functions from each experiment.

Usage:
    python run.py --exp1          # Feature discovery (H1, H3)
    python run.py --exp3          # Hindi-Urdu overlap (H4)
    python run.py --all           # Run all experiments
    python run.py --validate      # Validate setup only
"""

import argparse
import sys
import os
import random
from pathlib import Path

import numpy as np
import torch
from reproducibility import write_run_manifest


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
    # NOTE: exp4, exp6, exp16 are archived - don't validate them
    for exp_name in [
        "exp1_feature_discovery",
        "exp3_hindi_urdu_fixed",
        "exp5_hierarchical",
        "exp8_scaling_9b_low_resource",
        "exp9_layer_sweep_steering",
        "exp10_attribution_occlusion",
        "exp11_judge_calibration",
        "exp12_qa_degradation",
        "exp13_script_semantic_ablation",
        "exp14_language_agnostic_space",
        "exp15_steering_schedule_ablation",
        "exp18_typological_features",
        "exp19_crosslayer_causal",
        "exp20_training_freq_control",
        "exp21_family_separation",
        "exp22_feature_interpretation",
        "exp23_hierarchy_causal",
        "exp24_sae_detector",
        "exp25_family_causal",
    ]:
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


def _set_global_seeds(seed: int = 42):
    """Set RNG seeds once per process for reproducibility across experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass


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

# ARCHIVED: exp4 moved to archive/experiments_legacy/
# def run_exp4():
#     """Run Experiment 4: Language Steering Spillover Analysis."""
#     from experiments.exp4_spillover import main as exp4_main
#     exp4_main()

def run_exp5():
    """Run Experiment 5: Hierarchical Language Representation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Hierarchical Language Representation")
    print("=" * 60)
    print("Goal: Track language feature flow across layers")
    print("=" * 60 + "\n")

    from experiments.exp5_hierarchical import main as exp5_main
    exp5_main()

# ARCHIVED: exp6 moved to archive/experiments_legacy/
# def run_exp6():
#     """Run Experiment 6: Script vs Semantics Controls."""
#     from experiments.exp6_script_semantics_controls import main
#     main()

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


def run_exp10():
    """Run Experiment 10: Attribution via Feature Occlusion."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 10: Attribution via Feature Occlusion")
    print("=" * 60)
    print("Goal: Measure feature importance via occlusion")
    print("=" * 60 + "\n")

    from experiments.exp10_attribution_occlusion import main as exp10_main
    exp10_main()


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
    print("EXPERIMENT 14: Cross-Lingual Alignment")
    print("=" * 60)
    print("Goal: Find shared cross-lingual semantic features")
    print("=" * 60 + "\n")

    from experiments.exp14_language_agnostic_space import main as exp14_main
    exp14_main()


def run_exp15():
    """Run Experiment 15: Steering Schedule Ablation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 15: Steering Schedule Ablation")
    print("=" * 60)
    print("Goal: Compare steering schedules (constant, decay, prompt-only)")
    print("=" * 60 + "\n")

    from experiments.exp15_steering_schedule_ablation import main as exp15_main
    exp15_main()


# ARCHIVED: exp16 moved to archive/experiments_legacy/ (duplicate of exp22)
# def run_exp16():
#     """Run Experiment 16: Automated Feature Interpretation."""
#     from experiments.exp16_feature_interpretation import main as exp16_main
#     exp16_main()


def run_exp18():
    """Run Experiment 18: Typological Feature Analysis."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 18: Typological Feature Analysis")
    print("=" * 60)
    print("Goal: Test if clustering is due to typology (retroflex, SOV) vs family")
    print("=" * 60 + "\n")

    from experiments.exp18_typological_features import main as exp18_main
    exp18_main()


def run_exp19():
    """Run Experiment 19: Cross-Layer Causal Profiles."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 19: Cross-Layer Causal Profiles")
    print("=" * 60)
    print("Goal: Map causal importance across all layers via ablation")
    print("=" * 60 + "\n")

    from experiments.exp19_crosslayer_causal import main as exp19_main
    exp19_main()


def run_exp20():
    """Run Experiment 20: Training Data Frequency Control."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 20: Training Frequency Control")
    print("=" * 60)
    print("Goal: Test if clustering is a training frequency artifact")
    print("=" * 60 + "\n")

    from experiments.exp20_training_freq_control import main as exp20_main
    exp20_main()


def run_exp21():
    """Run Experiment 21: Indo-Aryan vs Dravidian Separation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 21: Family Separation Analysis")
    print("=" * 60)
    print("Goal: Test if Indo-Aryan and Dravidian are distinct sub-clusters")
    print("=" * 60 + "\n")

    from experiments.exp21_family_separation import main as exp21_main
    exp21_main()


def run_exp22():
    """Run Experiment 22: Systematic Feature Interpretation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 22: Feature Interpretation Pipeline")
    print("=" * 60)
    print("Goal: Auto-interpretability with monosemanticity metrics")
    print("=" * 60 + "\n")

    from experiments.exp22_feature_interpretation import main as exp22_main
    exp22_main()


def run_exp23():
    """Run Experiment 23: Causal Hierarchy Validation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 23: Hierarchy Causal Validation")
    print("=" * 60)
    print("Goal: Test Script->Language->Semantic hierarchy via ablation")
    print("=" * 60 + "\n")

    from experiments.exp23_hierarchy_causal import main as exp23_main
    exp23_main()


def run_exp24():
    """Run Experiment 24: SAE-Based Language Detector."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 24: SAE-Based Language Detector")
    print("=" * 60)
    print("Goal: Train classifier on SAE activations, test steering shift")
    print("=" * 60 + "\n")

    from experiments.exp24_sae_detector import main as exp24_main
    exp24_main()


def run_exp25():
    """Run Experiment 25: Family Feature Causal Ablation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 25: Family Feature Causality")
    print("=" * 60)
    print("Goal: Test if family-specific features are necessary/sufficient for steering")
    print("=" * 60 + "\n")

    from experiments.exp25_family_causal import main as exp25_main
    exp25_main()


def run_all():
    """Run all core experiments sequentially, then generate plots."""
    print("\n" + "=" * 60)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 60)

    # Publication-grade provenance: write a run manifest once per full run.
    try:
        manifest_path = write_run_manifest("results")
        print(f"[run] Wrote manifest to {manifest_path}")
    except Exception as e:
        print(f"[run] Warning: could not write manifest: {e}")
    
    # NOTE: Exp11 (judge calibration) is intentionally run before any
    # experiments that consume calibrated judge statistics (Exp9/10/12),
    # so that later runs can use bias-corrected scores when available.
    # Phase 1: Discovery
    # Phase 2: Calibration (run before experiments that use Gemini judge)
    # Phase 3: Core Steering
    # Phase 4: Causal Analysis (key experiments for paper)
    # Phase 5: Family & Typology
    # Phase 6: Validation
    # Phase 7: Interpretability
    # Optional experiments at end
    #
    # ARCHIVED (not run): exp4_spillover (covered by exp21),
    #                     exp6_script_semantics_controls (covered by exp13),
    #                     exp16_feature_interpretation (duplicate of exp22)
    experiments = [
        # Phase 1: Discovery
        ("exp1_feature_discovery", run_exp1),
        ("exp3_hindi_urdu_fixed", run_exp3),
        ("exp5_hierarchical", run_exp5),
        # Phase 2: Calibration
        ("exp11_judge_calibration", run_exp11),
        # Phase 3: Core Steering
        ("exp8_scaling_9b_low_resource", run_exp8),
        ("exp9_layer_sweep_steering", run_exp9),
        ("exp10_attribution_occlusion", run_exp10),
        # Phase 4: Causal Analysis (KEY)
        ("exp13_script_semantic_ablation", run_exp13),
        ("exp19_crosslayer_causal", run_exp19),
        ("exp23_hierarchy_causal", run_exp23),
        ("exp25_family_causal", run_exp25),
        # Phase 5: Family & Typology
        ("exp18_typological_features", run_exp18),
        ("exp21_family_separation", run_exp21),
        # Phase 6: Validation
        ("exp14_language_agnostic_space", run_exp14),
        ("exp24_sae_detector", run_exp24),
        # Phase 7: Interpretability
        ("exp22_feature_interpretation", run_exp22),
        # Optional
        ("exp12_qa_degradation", run_exp12),
        ("exp15_steering_schedule_ablation", run_exp15),
        ("exp20_training_freq_control", run_exp20),
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
        finally:
            # Best-effort cleanup to reduce peak VRAM across sequential runs.
            # This matters when running `python run.py --all` in one process on
            # shared GPUs where caching + fragmentation can accumulate.
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # IPC collect helps free cached allocator blocks in some setups.
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass
    
    # Generate plots at the end
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    try:
        import subprocess
        subprocess.run([sys.executable, "plots.py", "--results_dir", "results"], check=False)
        print("✓ Plot generation invoked (see results/figures)")
    except Exception as e:
        print(f"✗ Plot generation failed: {e}")

    # Generate text summary report
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    try:
        import subprocess
        subprocess.run([sys.executable, "summarize_results.py"], check=False)
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
  python run.py --exp3           # Hindi-Urdu overlap
  python run.py --all            # Run everything
        """
    )
    
    parser.add_argument("--validate", action="store_true", help="Validate setup only")
    parser.add_argument("--exp1", action="store_true", help="Run exp1: Feature Discovery")
    parser.add_argument("--exp3", action="store_true", help="Run exp3: Hindi-Urdu Overlap")
    # ARCHIVED: exp4, exp6 moved to archive/experiments_legacy/
    parser.add_argument("--exp5", action="store_true", help="Run exp5: Hierarchical Language Representation")
    parser.add_argument("--exp8", action="store_true", help="Run exp8: 2B vs 9B + Low-Resource Scaling")
    parser.add_argument("--exp9", action="store_true", help="Run exp9: Layer-wise Steering Sweep")
    parser.add_argument("--exp10", action="store_true", help="Run exp10: Occlusion-Based Attribution Steering")
    parser.add_argument("--exp11", action="store_true", help="Run exp11: Calibrated LLM-as-Judge Evaluation")
    parser.add_argument("--exp12", action="store_true", help="Run exp12: QA Degradation Under Steering")
    parser.add_argument("--exp13", action="store_true", help="Run exp13: Group Ablation for Script vs Semantic Features")
    parser.add_argument("--exp14", action="store_true", help="Run exp14: Cross-Lingual Alignment (Language-Agnostic Space)")
    parser.add_argument("--exp15", action="store_true", help="Run exp15: Steering Schedule Ablation")
    # ARCHIVED: exp16 moved to archive/experiments_legacy/ (use exp22 instead)
    parser.add_argument("--exp18", action="store_true", help="Run exp18: Typological Feature Analysis")
    parser.add_argument("--exp19", action="store_true", help="Run exp19: Cross-Layer Causal Profiles")
    parser.add_argument("--exp20", action="store_true", help="Run exp20: Training Frequency Control")
    parser.add_argument("--exp21", action="store_true", help="Run exp21: Indo-Aryan vs Dravidian Separation")
    parser.add_argument("--exp22", action="store_true", help="Run exp22: Feature Interpretation Pipeline")
    parser.add_argument("--exp23", action="store_true", help="Run exp23: Hierarchy Causal Validation")
    parser.add_argument("--exp24", action="store_true", help="Run exp24: SAE-Based Language Detector")
    parser.add_argument("--exp25", action="store_true", help="Run exp25: Family Feature Causality")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "--use_9b",
        action="store_true",
        help="Run experiments with Gemma 2 9B + 9B SAEs (writes *_9b.json outputs)",
    )
    
    args = parser.parse_args()

    # Global reproducibility baseline for every run invocation
    _set_global_seeds(42)

    # Global scaling toggle for all experiments in this process.
    if args.use_9b:
        os.environ["USE_9B"] = "1"
        print("[run] USE_9B=1 set: experiments will use Gemma 2 9B + 9B SAEs.")
    
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
    
    if args.exp3:
        run_exp3()

    # ARCHIVED: exp4, exp6 removed

    if args.exp5:
        run_exp5()

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

    # ARCHIVED: exp16 removed (use exp22 instead)

    if args.exp18:
        run_exp18()

    if args.exp19:
        run_exp19()

    if args.exp20:
        run_exp20()

    if args.exp21:
        run_exp21()

    if args.exp22:
        run_exp22()

    if args.exp23:
        run_exp23()

    if args.exp24:
        run_exp24()

    if args.exp25:
        run_exp25()

    if args.all:
        run_all()


if __name__ == "__main__":
    main()
