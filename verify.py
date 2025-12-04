"""Verification and debugging script for SAE multilingual experiments.

This script helps you:
1. Manually inspect features to verify they're language-specific
2. Debug the monolinguality steering failure
3. Verify Jaccard overlap calculations
4. Generate sanity-check statistics

Usage:
    python verify.py --mode inspect --feature_id 1234 --layer 13
    python verify.py --mode debug_mono
    python verify.py --mode check_overlap
    python verify.py --mode sanity_check
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter

# Optional imports
try:
    from config import MONOLINGUALITY_THRESHOLD, TARGET_LAYERS, LANGUAGES
    from data import load_flores
    from model import GemmaWithSAE
    HAS_PROJECT = True
except ImportError:
    HAS_PROJECT = False
    print("Warning: Project modules not found. Some features may not work.")


def inspect_feature(model, layer, feature_id, texts_by_lang, n_examples=5):
    """Manually inspect what a specific SAE feature responds to.
    
    This is the MOST IMPORTANT verification step. Look at the examples
    and ask: "Does this feature really capture Hindi-specific content?"
    """
    print(f"\n{'='*60}")
    print(f"FEATURE INSPECTION: Layer {layer}, Feature {feature_id}")
    print(f"{'='*60}")
    
    sae = model.load_sae(layer)
    
    for lang, texts in texts_by_lang.items():
        print(f"\n--- Language: {lang.upper()} ---")
        
        activations = []
        for text in texts[:min(50, len(texts))]:
            try:
                acts = model.get_sae_activations(text, layer)
                # Get max activation for this feature across all positions
                max_act = acts[:, feature_id].max().item()
                activations.append((max_act, text))
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
        
        if not activations:
            print("  No valid activations")
            continue
        
        # Statistics
        acts_only = [a[0] for a in activations]
        mean_act = np.mean(acts_only)
        max_act = np.max(acts_only)
        std_act = np.std(acts_only)
        nonzero_pct = sum(1 for a in acts_only if a > 0.01) / len(acts_only) * 100
        
        print(f"  Stats: mean={mean_act:.4f}, max={max_act:.4f}, std={std_act:.4f}")
        print(f"  Non-zero activations: {nonzero_pct:.1f}%")
        
        # Top activating examples
        activations.sort(key=lambda x: x[0], reverse=True)
        print(f"\n  Top {n_examples} activating examples:")
        for act, text in activations[:n_examples]:
            # Truncate text for display
            display_text = text[:60] + "..." if len(text) > 60 else text
            print(f"    [{act:.4f}] {display_text}")


def debug_monolinguality_steering(model, layer=13):
    """Debug why monolinguality-based steering failed.
    
    Hypotheses:
    1. Monolinguality features are "detectors" not "generators"
    2. Threshold is too strict/loose
    3. Feature selection is correct but steering direction is wrong
    """
    print(f"\n{'='*60}")
    print(f"DEBUGGING MONOLINGUALITY STEERING FAILURE")
    print(f"{'='*60}")
    
    # Load data
    flores = load_flores(max_samples=100)
    texts_hi = flores["hi"]
    texts_en = flores["en"]
    
    # Get monolinguality features
    from experiments.exp1_feature_discovery import compute_activation_rates, compute_monolinguality
    
    rates_by_lang = {}
    for lang, texts in flores.items():
        if len(texts) > 0:
            rates_by_lang[lang] = compute_activation_rates(model, texts[:100], layer)
    
    mono_scores = compute_monolinguality(rates_by_lang, "hi")
    
    # Analysis
    print(f"\n1. Monolinguality Score Distribution:")
    print(f"   Min: {mono_scores.min():.4f}")
    print(f"   Max: {mono_scores.max():.4f}")
    print(f"   Mean: {mono_scores.mean():.4f}")
    print(f"   Features with M > 3.0: {(mono_scores > 3.0).sum().item()}")
    print(f"   Features with M > 2.0: {(mono_scores > 2.0).sum().item()}")
    print(f"   Features with M > 1.5: {(mono_scores > 1.5).sum().item()}")
    
    # Get top monolinguality features
    top_k = 25
    _, top_mono_ids = mono_scores.topk(top_k)
    top_mono_ids = top_mono_ids.tolist()
    
    print(f"\n2. Top {top_k} Monolinguality Features:")
    for i, fid in enumerate(top_mono_ids[:5]):
        print(f"   Feature {fid}: M = {mono_scores[fid]:.4f}")
    
    # Check if these features actually activate for Hindi
    print(f"\n3. Do top mono features activate for Hindi?")
    sae = model.load_sae(layer)
    
    hi_activations = []
    en_activations = []
    
    for text in texts_hi[:20]:
        acts = model.get_sae_activations(text, layer)
        hi_activations.append(acts[:, top_mono_ids].max(dim=0).values)
    
    for text in texts_en[:20]:
        acts = model.get_sae_activations(text, layer)
        en_activations.append(acts[:, top_mono_ids].max(dim=0).values)
    
    hi_mean = torch.stack(hi_activations).mean(dim=0)
    en_mean = torch.stack(en_activations).mean(dim=0)
    
    print(f"   Hindi mean activation: {hi_mean.mean():.4f}")
    print(f"   English mean activation: {en_mean.mean():.4f}")
    print(f"   Ratio (Hindi/English): {(hi_mean.mean() / (en_mean.mean() + 1e-8)):.2f}x")
    
    # Now get activation-diff features for comparison
    print(f"\n4. Compare with Activation-Diff Features:")
    
    hi_acts_all = []
    en_acts_all = []
    
    for text in texts_hi[:50]:
        acts = model.get_sae_activations(text, layer)
        hi_acts_all.append(acts.mean(dim=0))
    
    for text in texts_en[:50]:
        acts = model.get_sae_activations(text, layer)
        en_acts_all.append(acts.mean(dim=0))
    
    hi_mean_all = torch.stack(hi_acts_all).mean(dim=0)
    en_mean_all = torch.stack(en_acts_all).mean(dim=0)
    
    act_diff = hi_mean_all - en_mean_all
    _, top_diff_ids = act_diff.topk(top_k)
    top_diff_ids = top_diff_ids.tolist()
    
    # Check overlap
    overlap = set(top_mono_ids) & set(top_diff_ids)
    print(f"   Overlap between mono and act-diff top-25: {len(overlap)} features")
    print(f"   Overlap percentage: {len(overlap)/top_k*100:.1f}%")
    
    if len(overlap) < 5:
        print("\n   ⚠️ LOW OVERLAP: Mono and act-diff select DIFFERENT features!")
        print("   This explains why mono-steering fails - wrong features selected.")
    
    # Hypothesis: Mono features might be "detectors" (fire when Hindi present)
    # but not "generators" (useful direction for steering toward Hindi)
    print(f"\n5. Hypothesis Check: Are mono features 'detectors' not 'generators'?")
    
    # Check if mono features have HIGH activation for Hindi, vs
    # act-diff features have HIGHER-THAN-ENGLISH activation
    mono_hi_ratio = (hi_mean.mean() / (en_mean.mean() + 1e-8)).item()
    
    diff_hi_activations = []
    diff_en_activations = []
    for text in texts_hi[:20]:
        acts = model.get_sae_activations(text, layer)
        diff_hi_activations.append(acts[:, top_diff_ids].max(dim=0).values)
    for text in texts_en[:20]:
        acts = model.get_sae_activations(text, layer)
        diff_en_activations.append(acts[:, top_diff_ids].max(dim=0).values)
    
    diff_hi_mean = torch.stack(diff_hi_activations).mean(dim=0)
    diff_en_mean = torch.stack(diff_en_activations).mean(dim=0)
    diff_ratio = (diff_hi_mean.mean() / (diff_en_mean.mean() + 1e-8)).item()
    
    print(f"   Mono features: Hindi/English ratio = {mono_hi_ratio:.2f}x")
    print(f"   ActDiff features: Hindi/English ratio = {diff_ratio:.2f}x")
    
    if mono_hi_ratio > diff_ratio:
        print("\n   ✓ Mono features ARE more Hindi-specific")
        print("   But they might not capture the EN→HI transformation direction!")
    else:
        print("\n   ⚠️ ActDiff features are actually more Hindi-specific")


def check_overlap_calculation():
    """Verify that Jaccard overlap is calculated correctly.
    
    Your results showed >100% overlap which is mathematically impossible.
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING JACCARD OVERLAP CALCULATION")
    print(f"{'='*60}")
    
    # Correct Jaccard formula
    print("\nCorrect Jaccard formula:")
    print("  J(A, B) = |A ∩ B| / |A ∪ B|")
    print("  Range: 0.0 to 1.0 (or 0% to 100%)")
    print("  CANNOT exceed 100%!")
    
    # Test with examples
    set_a = {1, 2, 3, 4, 5}
    set_b = {3, 4, 5, 6, 7}
    
    intersection = set_a & set_b
    union = set_a | set_b
    jaccard = len(intersection) / len(union)
    
    print(f"\nExample:")
    print(f"  Set A: {set_a}")
    print(f"  Set B: {set_b}")
    print(f"  Intersection: {intersection}")
    print(f"  Union: {union}")
    print(f"  Jaccard: {len(intersection)}/{len(union)} = {jaccard:.2f} ({jaccard*100:.0f}%)")
    
    # What could cause >100%?
    print("\n⚠️ Your results show >100% overlap. Possible bugs:")
    print("  1. Using feature COUNTS instead of SET intersection")
    print("  2. Dividing by min(|A|, |B|) instead of |A ∪ B|")
    print("  3. Including duplicate features in sets")
    print("  4. Counting features multiple times")
    
    # Show the buggy calculation that would give >100%
    print("\nBuggy calculation that gives >100%:")
    count_a = 8519  # Hindi features from your results
    count_b = 8546  # Urdu features
    buggy_ratio = count_a / count_b * 100
    print(f"  8519 / 8546 = {buggy_ratio:.1f}% (NOT Jaccard!)")
    
    print("\n✓ FIX: Ensure you're using set intersection, not counts:")
    print("""
    def correct_jaccard(features_a, features_b):
        set_a = set(features_a)
        set_b = set(features_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
    """)


def sanity_check(model=None, layer=13):
    """Run sanity checks on the experiment setup."""
    print(f"\n{'='*60}")
    print(f"SANITY CHECK")
    print(f"{'='*60}")
    
    # Check 1: SAE dimensions
    print("\n1. SAE Configuration:")
    print(f"   Expected features: 16384 (16k width)")
    print(f"   Expected hidden dim: 2304 (Gemma 2 2B)")
    
    if model is not None:
        sae = model.load_sae(layer)
        print(f"   Actual SAE features: {sae.cfg.d_sae}")
        print(f"   Actual hidden dim: {sae.cfg.d_in}")
        
        if sae.cfg.d_sae != 16384:
            print("   ⚠️ WARNING: Unexpected SAE width!")
    
    # Check 2: Dead features
    print("\n2. Dead Feature Analysis:")
    print("   Your results show ~15% dead features per language")
    print("   This is NORMAL for SAEs - Gemma Scope reports similar numbers")
    
    # Check 3: Sample sizes
    print("\n3. Sample Size Check:")
    print("   N_SAMPLES_DISCOVERY = 500 (reasonable for exploration)")
    print("   N_SAMPLES_EVAL = ? (should be ≥100 for statistical power)")
    
    # Check 4: Threshold
    print("\n4. Monolinguality Threshold:")
    print(f"   Current: M > 3.0")
    print("   M = 3.0 means feature is 3x more likely for target language")
    print("   Consider testing: M > 2.0, M > 5.0")
    
    # Check 5: Layer coverage
    print("\n5. Layer Coverage:")
    print(f"   Testing layers: 5, 8, 10, 13, 16, 20, 24")
    print("   Mid-range (40-60% of 26): layers 10-16")
    print("   Your peak at layer 24 suggests late-layer features are most specific")


def generate_verification_report(results_dir):
    """Generate a comprehensive verification report."""
    print(f"\n{'='*60}")
    print(f"VERIFICATION REPORT")
    print(f"{'='*60}")
    
    results_dir = Path(results_dir)
    
    # Load results
    results = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            results[json_file.stem] = json.load(f)
    
    print(f"\nFound {len(results)} result files")
    
    # Check for anomalies
    print("\nChecking for anomalies...")
    
    anomalies = []
    
    # Check exp2 (steering)
    if "exp2_steering_comparison" in results:
        exp2 = results["exp2_steering_comparison"]
        methods = exp2.get("methods", {})
        
        mono_best = 0
        for strength, data in methods.get("monolinguality", {}).items():
            if isinstance(data, dict):
                mono_best = max(mono_best, data.get("success_rate", 0))
        
        if mono_best == 0:
            anomalies.append("CRITICAL: Monolinguality steering has 0% success rate")
    
    # Check exp3 (overlap)
    if "exp3_hindi_urdu_overlap" in results:
        exp3 = results["exp3_hindi_urdu_overlap"]
        
        for layer_key, layer_data in exp3.items():
            if isinstance(layer_data, dict):
                overlap = layer_data.get("semantic_overlap", 0)
                if overlap > 1.0:
                    anomalies.append(f"BUG: {layer_key} has {overlap*100:.1f}% overlap (>100% impossible)")
    
    # Report anomalies
    if anomalies:
        print("\n⚠️ ANOMALIES DETECTED:")
        for a in anomalies:
            print(f"   - {a}")
    else:
        print("\n✓ No obvious anomalies found")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
1. FIX JACCARD CALCULATION: Use set intersection, not counts
   
2. DEBUG MONOLINGUALITY: Run the debug_mono mode to understand
   why it fails while activation-diff succeeds
   
3. INCREASE EVAL SAMPLES: 10 prompts is too few for statistical
   significance. Use at least 100 diverse prompts.
   
4. ADD CONFIDENCE INTERVALS: Run experiments 3x with different
   seeds and report mean ± std.
   
5. REFRAME H3: Your finding that late layers have MORE language
   features is actually interesting - it suggests a revision
   to the "Messy Middle" hypothesis.
""")


def main():
    parser = argparse.ArgumentParser(description='Verification and debugging tools')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['inspect', 'debug_mono', 'check_overlap', 
                                 'sanity_check', 'report'],
                        help='Verification mode')
    parser.add_argument('--feature_id', type=int, default=None,
                        help='Feature ID to inspect')
    parser.add_argument('--layer', type=int, default=13,
                        help='Layer to analyze')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory')
    
    args = parser.parse_args()
    
    if args.mode == 'check_overlap':
        check_overlap_calculation()
        return
    
    if args.mode == 'report':
        generate_verification_report(args.results_dir)
        return
    
    if args.mode == 'sanity_check':
        if HAS_PROJECT:
            print("Loading model for sanity check...")
            model = GemmaWithSAE()
            model.load_model()
            sanity_check(model, args.layer)
        else:
            sanity_check(None, args.layer)
        return
    
    # Modes that require model
    if not HAS_PROJECT:
        print("ERROR: Project modules required. Run from project directory.")
        return
    
    print("Loading model...")
    model = GemmaWithSAE()
    model.load_model()
    
    print("Loading data...")
    flores = load_flores(max_samples=100)
    
    if args.mode == 'inspect':
        if args.feature_id is None:
            # Inspect top monolinguality feature
            from experiments.exp1_feature_discovery import compute_activation_rates, compute_monolinguality
            rates = {}
            for lang, texts in flores.items():
                if len(texts) > 0:
                    rates[lang] = compute_activation_rates(model, texts[:50], args.layer)
            mono = compute_monolinguality(rates, "hi")
            args.feature_id = mono.argmax().item()
            print(f"Auto-selected top mono feature: {args.feature_id}")
        
        inspect_feature(model, args.layer, args.feature_id, flores)
    
    elif args.mode == 'debug_mono':
        debug_monolinguality_steering(model, args.layer)


if __name__ == "__main__":
    main()
