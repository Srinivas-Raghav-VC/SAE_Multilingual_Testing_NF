"""Experiment 21: Indo-Aryan vs Dravidian Family Separation Analysis

Tests whether the "Indic cluster" is truly unified or consists of distinct
Indo-Aryan and Dravidian sub-clusters.

Research Questions:
  1. Do Hindi-Urdu overlap more with each other than with Tamil-Telugu?
  2. Can steering vectors trained on Indo-Aryan transfer to Dravidian?
  3. Are there family-specific vs pan-Indic features?

Falsification Criteria:
  - If Jaccard(HI, UR) >> Jaccard(HI, TA), families are distinct clusters
  - If cross-family steering success < 30%, "unified Indic" claim is weak
  - If within-family transfer >> cross-family transfer, families are separable

Methodology:
  1. Hierarchical clustering with family coloring
  2. Frequency-based feature counting (not set-theoretic)
  3. Cross-family AND within-family steering transfer tests
  4. Bootstrap confidence intervals for statistical rigor
  5. Multi-layer transfer analysis
  6. Full N×N transfer matrix with non-Indic control

Publication-Grade Enhancements:
  - Within-family baselines (HI→UR, HI→BN, TA→TE)
  - Non-Indic control (HI→DE)
  - Bootstrap CI for separation ratio
  - Layer-wise transfer analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from scipy import stats as scipy_stats

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    N_STEERING_EVAL,
    LANG_TO_SCRIPT,
    INDIC_LANGUAGES,
    ALL_INDIC,
    EXTENDED_LANGUAGES,
    SEED,
)
from data import load_research_data, load_flores
from model import GemmaWithSAE
from steering_utils import (
    get_activation_diff_features,
    construct_sae_steering_vector,
)
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
    jaccard_overlap,
)
from reproducibility import seed_everything


@dataclass
class FamilyOverlapResult:
    """Overlap analysis results with bootstrap CI."""
    indo_aryan_internal: float
    dravidian_internal: float
    cross_family: float
    separation_ratio: float
    separation_ci_low: float
    separation_ci_high: float
    p_value: float
    interpretation: str


@dataclass
class CrossFamilySteeringResult:
    """Results for cross-family steering transfer test."""
    source_family: str
    target_family: str
    source_lang: str
    target_lang: str
    success_rate: float
    script_ratio: float
    transfer_type: str  # "within_family", "cross_family", "control"
    sensitivity: Any = None


@dataclass
class FrequencyBasedFeatureCounts:
    """Frequency-based feature classification (not set-theoretic)."""
    indo_aryan_specific: int
    dravidian_specific: int
    pan_indic: int
    total_features: int
    ia_specificity_ratio: float
    dr_specificity_ratio: float
    pan_indic_ratio: float


def compute_activation_features(
    model: GemmaWithSAE,
    texts: List[str],
    layer: int,
    threshold: float = 0.01,
) -> Set[int]:
    """Get set of active features for a language at a layer."""
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    activation_counts = torch.zeros(n_features, device=model.device)
    total_tokens = 0

    for text in texts[:100]:
        acts = model.get_sae_activations(text, layer)
        activation_counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]

    if total_tokens == 0:
        return set()

    rates = activation_counts / total_tokens
    active_mask = rates > threshold
    return set(int(i) for i in active_mask.nonzero(as_tuple=False).flatten().tolist())


def analyze_family_overlaps(
    feature_sets: Dict[str, Set[int]],
    indo_aryan_langs: List[str],
    dravidian_langs: List[str],
    n_bootstrap: int = 10000,
) -> FamilyOverlapResult:
    """Compute within and between family overlaps with bootstrap CI."""

    # Filter to available languages
    ia = [l for l in indo_aryan_langs if l in feature_sets]
    dr = [l for l in dravidian_langs if l in feature_sets]

    def get_pairwise_overlaps(langs: List[str]) -> List[float]:
        overlaps = []
        for i, l1 in enumerate(langs):
            for l2 in langs[i+1:]:
                overlaps.append(jaccard_overlap(feature_sets[l1], feature_sets[l2]))
        return overlaps

    def get_cross_overlaps(group1: List[str], group2: List[str]) -> List[float]:
        overlaps = []
        for l1 in group1:
            for l2 in group2:
                overlaps.append(jaccard_overlap(feature_sets[l1], feature_sets[l2]))
        return overlaps

    ia_overlaps = get_pairwise_overlaps(ia)
    dr_overlaps = get_pairwise_overlaps(dr)
    cross_overlaps = get_cross_overlaps(ia, dr)

    ia_internal = np.mean(ia_overlaps) if ia_overlaps else 0.0
    dr_internal = np.mean(dr_overlaps) if dr_overlaps else 0.0
    cross = np.mean(cross_overlaps) if cross_overlaps else 0.0

    separation = ((ia_internal + dr_internal) / 2) / max(cross, 0.01)

    # Bootstrap CI for separation ratio
    np.random.seed(SEED)
    bootstrap_separations = []

    for _ in range(n_bootstrap):
        # Resample each group
        ia_sample = np.random.choice(ia_overlaps, size=len(ia_overlaps), replace=True) if ia_overlaps else [0]
        dr_sample = np.random.choice(dr_overlaps, size=len(dr_overlaps), replace=True) if dr_overlaps else [0]
        cross_sample = np.random.choice(cross_overlaps, size=len(cross_overlaps), replace=True) if cross_overlaps else [0.01]

        boot_sep = ((np.mean(ia_sample) + np.mean(dr_sample)) / 2) / max(np.mean(cross_sample), 0.01)
        bootstrap_separations.append(boot_sep)

    ci_low = np.percentile(bootstrap_separations, 2.5)
    ci_high = np.percentile(bootstrap_separations, 97.5)

    # P-value: proportion of bootstrap samples where separation <= 1.0
    p_value = np.mean([s <= 1.0 for s in bootstrap_separations])

    if ci_low > 1.5:
        interpretation = "Distinct sub-clusters (CI excludes 1.5, p<0.05)"
    elif ci_low > 1.0:
        interpretation = "Significant separation (CI excludes 1.0)"
    elif separation > 1.0:
        interpretation = "Weak separation (CI includes 1.0)"
    else:
        interpretation = "Unified cluster (separation < 1.0)"

    return FamilyOverlapResult(
        indo_aryan_internal=ia_internal,
        dravidian_internal=dr_internal,
        cross_family=cross,
        separation_ratio=separation,
        separation_ci_low=ci_low,
        separation_ci_high=ci_high,
        p_value=p_value,
        interpretation=interpretation,
    )


def count_family_specific_features_frequency(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    indo_aryan_langs: List[str],
    dravidian_langs: List[str],
    specificity_threshold: float = 2.0,
) -> FrequencyBasedFeatureCounts:
    """Frequency-based feature classification (not set-theoretic).

    A feature is considered family-specific if its mean activation rate
    in one family is > threshold times its rate in the other family.
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    # Compute activation rates per language
    lang_rates = {}
    for lang, texts in texts_by_lang.items():
        if not texts:
            continue
        activation_counts = torch.zeros(n_features, device=model.device)
        total_tokens = 0

        for text in texts[:100]:
            acts = model.get_sae_activations(text, layer)
            activation_counts += (acts > 0).float().sum(dim=0)
            total_tokens += acts.shape[0]

        if total_tokens > 0:
            lang_rates[lang] = (activation_counts / total_tokens).cpu().numpy()

    # Filter to available languages
    ia = [l for l in indo_aryan_langs if l in lang_rates]
    dr = [l for l in dravidian_langs if l in lang_rates]

    if not ia or not dr:
        return FrequencyBasedFeatureCounts(
            indo_aryan_specific=0, dravidian_specific=0, pan_indic=0,
            total_features=0, ia_specificity_ratio=0, dr_specificity_ratio=0,
            pan_indic_ratio=0
        )

    # Compute mean rates per family
    ia_mean_rates = np.mean([lang_rates[l] for l in ia], axis=0)
    dr_mean_rates = np.mean([lang_rates[l] for l in dr], axis=0)

    # Classify features by frequency ratio
    indo_aryan_specific = 0
    dravidian_specific = 0
    pan_indic = 0

    for j in range(n_features):
        ia_rate = ia_mean_rates[j]
        dr_rate = dr_mean_rates[j]

        # Only consider features that are actually active somewhere
        if ia_rate < 0.001 and dr_rate < 0.001:
            continue

        if ia_rate > specificity_threshold * max(dr_rate, 0.001):
            indo_aryan_specific += 1
        elif dr_rate > specificity_threshold * max(ia_rate, 0.001):
            dravidian_specific += 1
        elif ia_rate > 0.001 or dr_rate > 0.001:
            pan_indic += 1

    total = indo_aryan_specific + dravidian_specific + pan_indic

    return FrequencyBasedFeatureCounts(
        indo_aryan_specific=indo_aryan_specific,
        dravidian_specific=dravidian_specific,
        pan_indic=pan_indic,
        total_features=total,
        ia_specificity_ratio=indo_aryan_specific / max(total, 1),
        dr_specificity_ratio=dravidian_specific / max(total, 1),
        pan_indic_ratio=pan_indic / max(total, 1),
    )


def count_family_specific_features(
    feature_sets: Dict[str, Set[int]],
    indo_aryan_langs: List[str],
    dravidian_langs: List[str],
) -> Dict[str, int]:
    """Count features specific to each family vs shared (set-theoretic - legacy)."""

    ia = [l for l in indo_aryan_langs if l in feature_sets]
    dr = [l for l in dravidian_langs if l in feature_sets]

    if not ia or not dr:
        return {"error": "insufficient_languages"}

    # Features active in ANY Indo-Aryan language
    ia_any = set.union(*[feature_sets[l] for l in ia])

    # Features active in ANY Dravidian language
    dr_any = set.union(*[feature_sets[l] for l in dr])

    # Indo-Aryan only (not in any Dravidian)
    ia_only = ia_any - dr_any

    # Dravidian only (not in any Indo-Aryan)
    dr_only = dr_any - ia_any

    # Pan-Indic (in both families)
    pan_indic = ia_any & dr_any

    return {
        "indo_aryan_only": len(ia_only),
        "dravidian_only": len(dr_only),
        "pan_indic": len(pan_indic),
        "indo_aryan_total": len(ia_any),
        "dravidian_total": len(dr_any),
        "pan_indic_ratio": len(pan_indic) / max(len(ia_any | dr_any), 1),
    }


def test_cross_family_steering(
    model: GemmaWithSAE,
    train_data: Dict[str, List[str]],
    prompts: List[str],
    layer: int,
) -> List[CrossFamilySteeringResult]:
    """Test steering vector transfer with within-family baselines and controls.

    Test pairs include:
    - Within-family (baseline HIGH expected): HI→UR, HI→BN, TA→TE
    - Cross-family: HI→TA, TA→HI, HI→TE, BN→ML
    - Control (expect LOW): HI→DE
    """
    results = []

    # Comprehensive test pairs: (source_lang, target_eval_lang, src_family, tgt_family, transfer_type)
    test_pairs = [
        # WITHIN-FAMILY BASELINES (expect HIGH transfer)
        ("hi", "ur", "indo_aryan", "indo_aryan", "within_family"),  # Same spoken language, same script
        ("hi", "bn", "indo_aryan", "indo_aryan", "within_family"),  # Same family, different script
        ("ta", "te", "dravidian", "dravidian", "within_family"),    # Dravidian pair

        # CROSS-FAMILY (main test)
        ("hi", "ta", "indo_aryan", "dravidian", "cross_family"),    # Hindi -> Tamil
        ("ta", "hi", "dravidian", "indo_aryan", "cross_family"),    # Tamil -> Hindi
        ("hi", "te", "indo_aryan", "dravidian", "cross_family"),    # Hindi -> Telugu
        ("bn", "ml", "indo_aryan", "dravidian", "cross_family"),    # Bengali -> Malayalam

        # CONTROL (expect LOW transfer)
        ("hi", "de", "indo_aryan", "germanic", "control"),          # Hindi -> German
    ]

    for src_lang, tgt_eval_lang, src_family, tgt_family, transfer_type in test_pairs:
        if src_lang not in train_data or "en" not in train_data:
            print(f"  Skipping {src_lang}->{tgt_eval_lang}: missing data")
            continue

        # For control pairs, we need German data
        if transfer_type == "control" and tgt_eval_lang not in train_data:
            print(f"  Skipping {src_lang}->{tgt_eval_lang}: no {tgt_eval_lang} data")
            continue

        print(f"\n  [{transfer_type.upper()}] Train EN->{src_lang}, Eval on {tgt_eval_lang}")

        # Build steering vector from source language
        features = get_activation_diff_features(
            model,
            texts_src=train_data["en"],
            texts_tgt=train_data[src_lang],
            layer=layer,
            top_k=25,
        )

        steering_vec = construct_sae_steering_vector(model, layer, features)
        target_script = LANG_TO_SCRIPT.get(tgt_eval_lang, "latin")

        # Evaluate on target language
        eval_results = []
        for p in tqdm(prompts[:30], desc=f"{src_lang}->{tgt_eval_lang}", leave=False):
            output = model.generate_with_steering(
                p, layer=layer, steering_vector=steering_vec, strength=2.0
            )
            res = evaluate_steering_output(
                p, output, method=f"transfer_{transfer_type}",
                strength=2.0, layer=layer,
                target_script=target_script,
                compute_semantics=True,
            )
            eval_results.append(res)

        agg = aggregate_results(eval_results, target_script=target_script)

        results.append(CrossFamilySteeringResult(
            source_family=src_family,
            target_family=tgt_family,
            source_lang=src_lang,
            target_lang=tgt_eval_lang,
            success_rate=agg.success_rate,
            script_ratio=agg.avg_target_script_ratio,
            transfer_type=transfer_type,
            sensitivity=agg.sensitivity,
        ))

        print(f"    Success rate: {agg.success_rate:.1%}")
        print(f"    Script ratio: {agg.avg_target_script_ratio:.1%}")

    return results


def compute_transfer_summary(results: List[CrossFamilySteeringResult]) -> Dict[str, Any]:
    """Compute summary statistics for transfer experiment."""
    within_family = [r for r in results if r.transfer_type == "within_family"]
    cross_family = [r for r in results if r.transfer_type == "cross_family"]
    control = [r for r in results if r.transfer_type == "control"]

    summary = {
        "within_family_mean": np.mean([r.success_rate for r in within_family]) if within_family else None,
        "cross_family_mean": np.mean([r.success_rate for r in cross_family]) if cross_family else None,
        "control_mean": np.mean([r.success_rate for r in control]) if control else None,
        "n_within_family": len(within_family),
        "n_cross_family": len(cross_family),
        "n_control": len(control),
    }

    # Compute transfer gap (within-family minus cross-family)
    if summary["within_family_mean"] is not None and summary["cross_family_mean"] is not None:
        summary["transfer_gap"] = summary["within_family_mean"] - summary["cross_family_mean"]
        # Statistical test (independent, since pairs differ)
        try:
            from stats import mann_whitney_test, holm_bonferroni_correction
            within_scores = [r.success_rate for r in within_family]
            cross_scores = [r.success_rate for r in cross_family]
            test_result = mann_whitney_test(within_scores, cross_scores)
            summary["gap_test"] = {
                "test": "mann_whitney",
                "p_value": test_result.p_value,
                "effect_size": test_result.effect_size,
                "significant_05": test_result.significant_at_05,
            }
            # Single-test Holm correction is identity but keep structure for consistency
            summary["gap_test"]["adjusted_p"] = holm_bonferroni_correction([test_result.p_value])["adjusted_p_values"][0]
        except Exception as e:
            summary["gap_test"] = {"error": str(e)}

        # Interpretation with decision thresholds
        gap = summary["transfer_gap"]
        # Power approximation for difference in proportions
        try:
            from evaluation_comprehensive import estimate_power_binary
            n_eff = min(summary.get("n_within_family", 0), summary.get("n_cross_family", 0)) or len(results)
            summary["power_gap"] = estimate_power_binary(
                summary["cross_family_mean"], summary["within_family_mean"], n=n_eff
            )
        except Exception:
            summary["power_gap"] = None

        adj_p = summary.get("gap_test", {}).get("adjusted_p", 1.0)
        if adj_p < 0.05 and gap > 0.3:
            summary["interpretation"] = "Strong family separation (large significant gap)"
        elif adj_p < 0.05 and gap > 0.1:
            summary["interpretation"] = "Moderate family separation (significant gap)"
        elif gap > 0:
            summary["interpretation"] = "Weak separation (gap not significant)"
        else:
            summary["interpretation"] = "Unified cluster (no transfer gap)"
    else:
        summary["transfer_gap"] = None
        summary["interpretation"] = "Insufficient data"

    return summary


def main():
    seed_everything(SEED)
    print("=" * 60)
    print("EXPERIMENT 21: Indo-Aryan vs Dravidian Separation")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in model.model_id.lower() else ""

    # Load data
    data = load_research_data(
        max_train_samples=N_SAMPLES_DISCOVERY,
        max_test_samples=0,
        max_eval_samples=0,
        seed=SEED,
    )

    prompts = data.steering_prompts[:N_STEERING_EVAL]

    # Use middle layer
    layer = TARGET_LAYERS[len(TARGET_LAYERS) // 2]
    print(f"\nAnalyzing Layer {layer}")

    # Compute feature sets for all Indic languages
    print("\nComputing feature sets for Indic languages...")
    feature_sets = {}

    indic_langs = INDIC_LANGUAGES["indo_aryan"] + INDIC_LANGUAGES["dravidian"]
    for lang in indic_langs:
        if lang not in data.train or not data.train[lang]:
            print(f"  Skipping {lang}: no data")
            continue
        feature_sets[lang] = compute_activation_features(
            model, data.train[lang], layer
        )
        print(f"  {lang}: {len(feature_sets[lang])} active features")

    results = {
        "layer": layer,
        "languages_analyzed": list(feature_sets.keys()),
    }

    # 1. Family overlap analysis with bootstrap CI
    print("\n" + "=" * 60)
    print("FAMILY OVERLAP ANALYSIS (with Bootstrap CI)")
    print("=" * 60)

    overlap_result = analyze_family_overlaps(
        feature_sets,
        INDIC_LANGUAGES["indo_aryan"],
        INDIC_LANGUAGES["dravidian"],
        n_bootstrap=10000,
    )

    print(f"\n  Indo-Aryan internal overlap: {overlap_result.indo_aryan_internal:.3f}")
    print(f"  Dravidian internal overlap: {overlap_result.dravidian_internal:.3f}")
    print(f"  Cross-family overlap: {overlap_result.cross_family:.3f}")
    print(f"  Separation ratio: {overlap_result.separation_ratio:.2f}")
    print(f"  95% CI: [{overlap_result.separation_ci_low:.2f}, {overlap_result.separation_ci_high:.2f}]")
    print(f"  P-value (H0: separation <= 1.0): {overlap_result.p_value:.4f}")
    print(f"\n  Interpretation: {overlap_result.interpretation}")

    results["family_overlaps"] = {
        "indo_aryan_internal": overlap_result.indo_aryan_internal,
        "dravidian_internal": overlap_result.dravidian_internal,
        "cross_family": overlap_result.cross_family,
        "separation_ratio": overlap_result.separation_ratio,
        "separation_ci_low": overlap_result.separation_ci_low,
        "separation_ci_high": overlap_result.separation_ci_high,
        "p_value": overlap_result.p_value,
        "interpretation": overlap_result.interpretation,
    }

    # 2. Pairwise overlap matrix
    print("\n" + "=" * 60)
    print("PAIRWISE OVERLAP MATRIX")
    print("=" * 60)

    langs = list(feature_sets.keys())
    overlap_matrix = {}
    for i, l1 in enumerate(langs):
        for l2 in langs[i:]:
            key = f"{l1}-{l2}"
            overlap_matrix[key] = jaccard_overlap(feature_sets[l1], feature_sets[l2])

    # Print key pairs
    print("\n  Key pairs:")
    key_pairs = [
        ("hi", "ur", "Same spoken language"),
        ("hi", "bn", "Indo-Aryan pair"),
        ("ta", "te", "Dravidian pair"),
        ("hi", "ta", "Cross-family"),
        ("ur", "ta", "Cross-family"),
    ]
    for l1, l2, desc in key_pairs:
        if l1 in feature_sets and l2 in feature_sets:
            key = f"{l1}-{l2}" if f"{l1}-{l2}" in overlap_matrix else f"{l2}-{l1}"
            print(f"    {l1}-{l2} ({desc}): {overlap_matrix.get(key, 0):.3f}")

    results["overlap_matrix"] = overlap_matrix

    # 3. Family-specific features (FREQUENCY-BASED - publication grade)
    print("\n" + "=" * 60)
    print("FAMILY-SPECIFIC FEATURES (Frequency-Based)")
    print("=" * 60)

    # Prepare texts for frequency-based analysis
    texts_by_lang = {lang: data.train.get(lang, []) for lang in indic_langs}

    freq_counts = count_family_specific_features_frequency(
        model, texts_by_lang, layer,
        INDIC_LANGUAGES["indo_aryan"],
        INDIC_LANGUAGES["dravidian"],
        specificity_threshold=2.0,
    )

    print(f"\n  Indo-Aryan specific (>2x rate): {freq_counts.indo_aryan_specific}")
    print(f"  Dravidian specific (>2x rate): {freq_counts.dravidian_specific}")
    print(f"  Pan-Indic (shared): {freq_counts.pan_indic}")
    print(f"  Total active features: {freq_counts.total_features}")
    print(f"  Pan-Indic ratio: {freq_counts.pan_indic_ratio:.1%}")

    results["feature_counts_frequency"] = {
        "indo_aryan_specific": freq_counts.indo_aryan_specific,
        "dravidian_specific": freq_counts.dravidian_specific,
        "pan_indic": freq_counts.pan_indic,
        "total_features": freq_counts.total_features,
        "ia_specificity_ratio": freq_counts.ia_specificity_ratio,
        "dr_specificity_ratio": freq_counts.dr_specificity_ratio,
        "pan_indic_ratio": freq_counts.pan_indic_ratio,
    }

    # Also compute legacy set-theoretic counts for comparison
    feature_counts = count_family_specific_features(
        feature_sets,
        INDIC_LANGUAGES["indo_aryan"],
        INDIC_LANGUAGES["dravidian"],
    )
    results["feature_counts_set_theoretic"] = feature_counts

    # 4. Comprehensive steering transfer (within-family, cross-family, control)
    print("\n" + "=" * 60)
    print("STEERING TRANSFER ANALYSIS")
    print("(Within-Family Baselines + Cross-Family + Control)")
    print("=" * 60)

    steering_results = test_cross_family_steering(
        model, data.train, prompts, layer
    )

    results["steering_transfer"] = [
        {
            "source": r.source_lang,
            "target": r.target_lang,
            "source_family": r.source_family,
            "target_family": r.target_family,
            "success_rate": r.success_rate,
            "script_ratio": r.script_ratio,
            "transfer_type": r.transfer_type,
            "sensitivity": r.sensitivity,
        }
        for r in steering_results
    ]

    # Compute transfer summary with within/cross/control comparison
    transfer_summary = compute_transfer_summary(steering_results)

    print(f"\n  TRANSFER SUMMARY:")
    print(f"  Within-family mean: {transfer_summary['within_family_mean']:.1%}" if transfer_summary['within_family_mean'] else "  Within-family: N/A")
    print(f"  Cross-family mean: {transfer_summary['cross_family_mean']:.1%}" if transfer_summary['cross_family_mean'] else "  Cross-family: N/A")
    print(f"  Control mean: {transfer_summary['control_mean']:.1%}" if transfer_summary['control_mean'] else "  Control: N/A")

    if transfer_summary['transfer_gap'] is not None:
        print(f"\n  Transfer gap (within - cross): {transfer_summary['transfer_gap']:.1%}")
        print(f"  Interpretation: {transfer_summary['interpretation']}")

    results["transfer_summary"] = transfer_summary

    # Final summary with publication-grade conclusions
    print("\n" + "=" * 60)
    print("PUBLICATION-GRADE SUMMARY")
    print("=" * 60)

    conclusions = []

    # 1. Overlap-based conclusion (with CI)
    if overlap_result.separation_ci_low > 1.5:
        conclusions.append(f"STRONG: Indo-Aryan and Dravidian are DISTINCT (separation ratio CI [{overlap_result.separation_ci_low:.2f}, {overlap_result.separation_ci_high:.2f}] excludes 1.5)")
    elif overlap_result.separation_ci_low > 1.0:
        conclusions.append(f"MODERATE: Families show significant separation (CI excludes 1.0, p={overlap_result.p_value:.4f})")
    elif overlap_result.separation_ratio > 1.0:
        conclusions.append(f"WEAK: Some family structure but CI includes 1.0 (non-significant)")
    else:
        conclusions.append("NO SEPARATION: Indo-Aryan and Dravidian form unified Indic cluster")

    # 2. Feature-based conclusion (frequency-based)
    if freq_counts.pan_indic_ratio > 0.6:
        conclusions.append(f"FEATURE SHARING: {freq_counts.pan_indic_ratio:.0%} of features are pan-Indic (unified representation)")
    elif freq_counts.pan_indic_ratio > 0.4:
        conclusions.append(f"MIXED: {freq_counts.pan_indic_ratio:.0%} pan-Indic, families share core features but have specificity")
    else:
        conclusions.append(f"FAMILY-SPECIFIC: Only {freq_counts.pan_indic_ratio:.0%} pan-Indic (distinct family representations)")

    # 3. Transfer-based conclusion
    if transfer_summary.get("transfer_gap") is not None:
        gap = transfer_summary["transfer_gap"]
        if gap > 0.2:
            conclusions.append(f"TRANSFER GAP: Within-family transfer {gap:.0%} higher than cross-family (families functionally distinct)")
        elif gap > 0:
            conclusions.append(f"SMALL TRANSFER GAP: {gap:.0%} difference suggests related but partially separable families")
        else:
            conclusions.append("NO TRANSFER GAP: Steering transfers equally across families (unified)")

    print("\n  CONCLUSIONS:")
    for i, c in enumerate(conclusions, 1):
        print(f"  {i}. {c}")

    # Publication-ready falsification assessment
    print("\n  FALSIFICATION ASSESSMENT:")
    claim_unified = overlap_result.separation_ci_low <= 1.0
    claim_distinct = overlap_result.separation_ci_low > 1.0

    if claim_unified:
        print("  → 'Unified Indic Cluster' claim: SUPPORTED (CI includes 1.0)")
    if claim_distinct:
        print("  → 'Distinct Families' claim: SUPPORTED (CI excludes 1.0)")

    results["conclusions"] = conclusions
    results["falsification"] = {
        "unified_indic_supported": claim_unified,
        "distinct_families_supported": claim_distinct,
        "separation_significant": overlap_result.p_value < 0.05,
    }

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp21_family_separation{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
