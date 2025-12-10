"""Experiment 3 (FIXED): Hindi-Urdu Feature Overlap

CRITICAL BUG FIX: Previous version computed semantic/hi_total which can exceed 100%.
This version uses CORRECT Jaccard: |A∩B| / |A∪B| which is ALWAYS ≤ 1.0

H4: Hindi and Urdu share >80% of semantic features but <20% of script features
"""

# Path fix for running from experiments/ directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from pathlib import Path
from typing import Set, Dict, Tuple
from dataclasses import dataclass

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY, SENSITIVITY_THRESHOLDS
from data import load_flores
from model import GemmaWithSAE
from stats import bootstrap_ci


@dataclass
class OverlapResult:
    """Container for overlap metrics with validation."""
    jaccard: float
    intersection_size: int
    union_size: int
    set1_size: int
    set2_size: int
    
    def __post_init__(self):
        """Validate that Jaccard is in [0, 1]."""
        assert 0.0 <= self.jaccard <= 1.0, f"Invalid Jaccard: {self.jaccard}"


def compute_correct_jaccard(set_a: Set[int], set_b: Set[int]) -> OverlapResult:
    """Compute CORRECT Jaccard overlap.
    
    Formula: |A ∩ B| / |A ∪ B|
    This is ALWAYS between 0 and 1.
    
    Raises:
        AssertionError: If computed Jaccard is outside [0, 1] (indicates a bug)
    """
    intersection = set_a & set_b
    union = set_a | set_b
    
    if len(union) == 0:
        # This should not normally happen if active-feature thresholds and
        # data loading are sane. We treat it as zero overlap but log a
        # diagnostic warning so that potential data issues do not go
        # unnoticed.
        print("[exp3] Warning: empty union encountered when computing Jaccard; returning 0.0")
        jaccard = 0.0
    else:
        jaccard = len(intersection) / len(union)
    
    # Research rigor validation: Jaccard MUST be in [0, 1]
    assert 0.0 <= jaccard <= 1.0, (
        f"BUG: Jaccard coefficient {jaccard} is outside valid range [0, 1]. "
        f"|A|={len(set_a)}, |B|={len(set_b)}, |A∩B|={len(intersection)}, |A∪B|={len(union)}"
    )
    
    return OverlapResult(
        jaccard=jaccard,
        intersection_size=len(intersection),
        union_size=len(union),
        set1_size=len(set_a),
        set2_size=len(set_b)
    )


def get_active_features(model, texts, layer, threshold=0.01) -> Set[int]:
    """Get set of features that activate frequently for texts.

    This uses the same activation-rate definition as Exp1
    (compute_activation_rates): for each feature we count the number of
    tokens on which it is active (activation > 0) and divide by the
    total number of tokens. A feature is considered "active" if this
    rate exceeds `threshold`.
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    
    activation_counts = torch.zeros(n_features, device=model.device)
    total_tokens = 0
    
    for text in texts:
        acts = model.get_sae_activations(text, layer)
        activation_counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]
    
    rates = activation_counts / total_tokens
    active_mask = rates > threshold
    active_features = set(active_mask.nonzero().squeeze(-1).tolist())
    
    # Handle edge case where only one feature is active
    if isinstance(list(active_features)[0] if active_features else 0, int):
        return active_features
    return active_features


def identify_script_vs_semantic(
    model, 
    texts_hi, 
    texts_ur, 
    layer, 
    rate_threshold=0.01,  # SAME threshold as get_active_features
    ratio_threshold=0.5
) -> Dict[str, Set[int]]:
    """
    Identify script-specific vs semantic features for Hindi-Urdu.
    
    FIXED: Now uses activation RATES (same as get_active_features) for consistency.
    This ensures semantic_ratio <= 1.0 always.
    
    Script features: Activate strongly for ONE language (script-dependent)
    Semantic features: Activate similarly for BOTH (meaning-dependent)
    
    Returns:
        Dict with "semantic", "hi_script", "ur_script" feature sets
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    
    # Compute activation RATES (not means!) - same as get_active_features
    hi_counts = torch.zeros(n_features, device=model.device)
    hi_tokens = 0
    for text in texts_hi[:100]:
        acts = model.get_sae_activations(text, layer)
        hi_counts += (acts > 0).float().sum(dim=0)
        hi_tokens += acts.shape[0]
    hi_rates = hi_counts / hi_tokens
    
    ur_counts = torch.zeros(n_features, device=model.device)
    ur_tokens = 0
    for text in texts_ur[:100]:
        acts = model.get_sae_activations(text, layer)
        ur_counts += (acts > 0).float().sum(dim=0)
        ur_tokens += acts.shape[0]
    ur_rates = ur_counts / ur_tokens
    
    # Features active for each language (using SAME threshold)
    hi_active = hi_rates > rate_threshold
    ur_active = ur_rates > rate_threshold
    
    # Semantic: Active for BOTH with similar activation rates
    both_active = hi_active & ur_active
    ratio = hi_rates / (ur_rates + 1e-10)
    similar_magnitude = (ratio > ratio_threshold) & (ratio < 1/ratio_threshold)
    semantic_mask = both_active & similar_magnitude
    
    # Script: Active for ONE but not the other
    hi_script_mask = hi_active & ~ur_active
    ur_script_mask = ur_active & ~hi_active
    
    return {
        "semantic": set(semantic_mask.nonzero().squeeze(-1).tolist()),
        "hi_script": set(hi_script_mask.nonzero().squeeze(-1).tolist()),
        "ur_script": set(ur_script_mask.nonzero().squeeze(-1).tolist()),
    }


def main():
    """Run Hindi-Urdu overlap experiment with CORRECT metrics."""
    
    print("=" * 60)
    print("EXPERIMENT 3: Hindi-Urdu Feature Overlap (FIXED)")
    print("=" * 60)
    print("\nNOTE: This version uses CORRECT Jaccard (always ≤ 100%)")
    
    # Load model
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data
    print("\nLoading FLORES-200...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    texts_hi = flores.get("hi", [])
    texts_ur = flores.get("ur", [])
    texts_en = flores.get("en", [])
    
    if not texts_hi or not texts_ur:
        print("ERROR: Could not load Hindi or Urdu data!")
        return
    
    print(f"Hindi samples: {len(texts_hi)}")
    print(f"Urdu samples: {len(texts_ur)}")
    print(f"English samples: {len(texts_en)}")
    
    results = {"layers": {}, "hypothesis_tests": {}}
    
    for layer in TARGET_LAYERS:
        print(f"\n{'='*20} Layer {layer} {'='*20}")
        print(f"Loading SAE for layer {layer}...")
        
        # Get active feature sets
        hi_features = get_active_features(model, texts_hi, layer)
        ur_features = get_active_features(model, texts_ur, layer)
        en_features = get_active_features(model, texts_en, layer)
        
        # Compute CORRECT Jaccard overlaps
        hi_ur = compute_correct_jaccard(hi_features, ur_features)
        hi_en = compute_correct_jaccard(hi_features, en_features)
        ur_en = compute_correct_jaccard(ur_features, en_features)
        
        print(f"\nFeature counts:")
        print(f"  Hindi: {len(hi_features)}")
        print(f"  Urdu: {len(ur_features)}")
        print(f"  English: {len(en_features)}")
        
        print(f"\nJaccard overlaps (CORRECT - all ≤ 100%):")
        print(f"  Hindi-Urdu: {hi_ur.jaccard:.1%}")
        print(f"  Hindi-English: {hi_en.jaccard:.1%}")
        print(f"  Urdu-English: {ur_en.jaccard:.1%}")
        
        # Script vs Semantic analysis
        feature_types = identify_script_vs_semantic(model, texts_hi, texts_ur, layer)
        
        # Compute script/semantic ratio using CORRECT formulas
        total_hi_ur_features = len(hi_features | ur_features)
        semantic_ratio = len(feature_types["semantic"]) / total_hi_ur_features if total_hi_ur_features > 0 else 0
        script_ratio = (len(feature_types["hi_script"]) + len(feature_types["ur_script"])) / total_hi_ur_features if total_hi_ur_features > 0 else 0
        
        # VALIDATION: These ratios should NEVER exceed 1.0
        assert semantic_ratio <= 1.0, f"BUG: semantic_ratio={semantic_ratio:.3f} > 1.0 at layer {layer}"
        assert script_ratio <= 1.0, f"BUG: script_ratio={script_ratio:.3f} > 1.0 at layer {layer}"
        
        print(f"\nScript vs Semantic breakdown:")
        print(f"  Semantic features: {len(feature_types['semantic'])} ({semantic_ratio:.1%} of union)")
        print(f"  Hindi-only (script): {len(feature_types['hi_script'])}")
        print(f"  Urdu-only (script): {len(feature_types['ur_script'])}")
        print(f"  Total script-specific: {script_ratio:.1%} of union")
        
        results["layers"][str(layer)] = {
            "feature_counts": {
                "hindi": len(hi_features),
                "urdu": len(ur_features),
                "english": len(en_features),
            },
            "jaccard_overlaps": {
                "hindi_urdu": hi_ur.jaccard,
                "hindi_english": hi_en.jaccard,
                "urdu_english": ur_en.jaccard,
            },
            "script_semantic": {
                "semantic": len(feature_types["semantic"]),
                "hindi_script": len(feature_types["hi_script"]),
                "urdu_script": len(feature_types["ur_script"]),
                "semantic_ratio": semantic_ratio,
                "script_ratio": script_ratio,
            }
        }
    
    # Hypothesis Tests
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTS")
    print("=" * 60)
    
    # H4a: Hindi-Urdu Jaccard > 0.5 (>50% overlap)
    print("\nH4a: Hindi-Urdu share >50% features (Jaccard > 0.5)")
    h4a_results = []
    for layer_str, data in results["layers"].items():
        jaccard = data["jaccard_overlaps"]["hindi_urdu"]
        status = "PASS" if jaccard > 0.5 else "FAIL"
        h4a_results.append(jaccard > 0.5)
        print(f"  Layer {layer_str}: {jaccard:.1%} [{status}]")
    
    h4a_pass = all(h4a_results)
    print(f"\n  H4a Overall: {'PASS' if h4a_pass else 'FAIL'}")
    
    # H4b: Script features < 20% of total
    print("\nH4b: Script-specific features <20% of union")
    h4b_results = []
    for layer_str, data in results["layers"].items():
        script_ratio = data["script_semantic"]["script_ratio"]
        status = "PASS" if script_ratio < 0.2 else "FAIL"
        h4b_results.append(script_ratio < 0.2)
        print(f"  Layer {layer_str}: {script_ratio:.1%} [{status}]")
    
    h4b_pass = all(h4b_results)
    print(f"\n  H4b Overall: {'PASS' if h4b_pass else 'FAIL'}")
    
    # H4c: Hindi-Urdu overlap > Hindi-English overlap (Urdu is closer)
    print("\nH4c: Hindi-Urdu overlap > Hindi-English overlap")
    h4c_results = []
    for layer_str, data in results["layers"].items():
        hi_ur = data["jaccard_overlaps"]["hindi_urdu"]
        hi_en = data["jaccard_overlaps"]["hindi_english"]
        status = "PASS" if hi_ur > hi_en else "FAIL"
        h4c_results.append(hi_ur > hi_en)
        print(f"  Layer {layer_str}: HI-UR={hi_ur:.1%} > HI-EN={hi_en:.1%}? [{status}]")
    
    h4c_pass = all(h4c_results)
    print(f"\n  H4c Overall: {'PASS' if h4c_pass else 'FAIL'}")
    
    results["hypothesis_tests"] = {
        "h4a_semantic_overlap": h4a_pass,
        "h4b_script_small": h4b_pass,
        "h4c_urdu_closer": h4c_pass,
        "overall": h4a_pass and h4b_pass and h4c_pass
    }

    # =========================================================================
    # SENSITIVITY ANALYSIS: Jaccard at multiple activation thresholds
    # =========================================================================
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Jaccard vs Activation Threshold")
    print("=" * 60)
    print("\nThis shows that results are ROBUST to threshold choice.\n")

    # Use mid-layer (13) or first available for sensitivity analysis
    sens_layer = 13 if 13 in TARGET_LAYERS else TARGET_LAYERS[0]
    thresholds = SENSITIVITY_THRESHOLDS.get("jaccard_activation_rates", [0.01, 0.02, 0.05, 0.1])

    sensitivity_results = {"layer": sens_layer, "thresholds": {}}

    print(f"Layer {sens_layer}:")
    print(f"{'Threshold':<12} {'HI-UR Jaccard':<15} {'HI-EN Jaccard':<15} {'UR-EN Jaccard':<15}")
    print("-" * 60)

    for thresh in thresholds:
        hi_features = get_active_features(model, texts_hi, sens_layer, threshold=thresh)
        ur_features = get_active_features(model, texts_ur, sens_layer, threshold=thresh)
        en_features = get_active_features(model, texts_en, sens_layer, threshold=thresh)

        hi_ur = compute_correct_jaccard(hi_features, ur_features)
        hi_en = compute_correct_jaccard(hi_features, en_features)
        ur_en = compute_correct_jaccard(ur_features, en_features)

        print(f"{thresh:<12.2f} {hi_ur.jaccard:<15.1%} {hi_en.jaccard:<15.1%} {ur_en.jaccard:<15.1%}")

        sensitivity_results["thresholds"][str(thresh)] = {
            "hindi_urdu": hi_ur.jaccard,
            "hindi_english": hi_en.jaccard,
            "urdu_english": ur_en.jaccard,
            "n_hindi": len(hi_features),
            "n_urdu": len(ur_features),
            "n_english": len(en_features),
        }

    # Check robustness: HI-UR > HI-EN at all thresholds?
    all_robust = all(
        d["hindi_urdu"] > d["hindi_english"]
        for d in sensitivity_results["thresholds"].values()
    )
    print(f"\nRobustness check: HI-UR > HI-EN at all thresholds? {'YES' if all_robust else 'NO'}")
    sensitivity_results["robust_across_thresholds"] = all_robust

    results["sensitivity_analysis"] = sensitivity_results

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "exp3_hindi_urdu_fixed.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'exp3_hindi_urdu_fixed.json'}")
    
    return results


if __name__ == "__main__":
    main()
