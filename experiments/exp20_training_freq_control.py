"""Experiment 20: Training Data Frequency Control

Tests whether the observed "Indic cluster" is an artifact of Gemma's training
data distribution rather than genuine linguistic structure.

Research Questions:
  1. Is feature count correlated with training data frequency (inverse perplexity)?
  2. Do low-resource languages (Malayalam) cluster despite limited training data?
  3. Does perplexity explain feature overlap better than linguistic family?

Falsification Criteria:
  - If perplexity explains feature counts better than family, clustering is
    a frequency artifact
  - If low-resource Malayalam still clusters with Dravidian, frequency is
    not the primary driver

Methodology:
  1. Estimate Gemma training frequency via perplexity (lower = more training data)
  2. Correlate n_features with 1/perplexity
  3. Test if Malayalam (low-resource) clusters with other Dravidian languages
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple
from tqdm import tqdm

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    INDIC_LANGUAGES,
    ALL_INDIC,
    EXTENDED_LANGUAGES,
    SEED,
)
from data import load_flores
from model import GemmaWithSAE
from evaluation_comprehensive import jaccard_overlap
from reproducibility import seed_everything
from stats import holm_bonferroni_correction, mann_whitney_test
from evaluation_comprehensive import estimate_power_binary, estimate_power_independent


@dataclass
class LanguageFrequencyEstimate:
    """Estimated training frequency for a language."""
    lang: str
    perplexity: float
    inverse_perplexity: float  # Proxy for training frequency
    n_active_features: int


def compute_perplexity(
    model: GemmaWithSAE,
    texts: List[str],
    max_texts: int = 50,
) -> float:
    """Compute average perplexity on texts as proxy for training frequency."""

    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts[:max_texts], desc="Perplexity", leave=False):
        inputs = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

        n_tokens = inputs["input_ids"].shape[1]
        total_loss += loss * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def compute_active_features(
    model: GemmaWithSAE,
    texts: List[str],
    layer: int,
    threshold: float = 0.01,
) -> Set[int]:
    """Get set of active features for a language."""
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


def analyze_frequency_correlation(
    estimates: List[LanguageFrequencyEstimate],
) -> Dict[str, float]:
    """Analyze correlation between frequency and feature counts."""

    if len(estimates) < 3:
        return {"error": "insufficient_data"}

    inv_ppl = np.array([e.inverse_perplexity for e in estimates])
    n_feats = np.array([e.n_active_features for e in estimates])

    # Pearson correlation
    corr = np.corrcoef(inv_ppl, n_feats)[0, 1]
    r_squared = corr ** 2

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    spearman_corr, p_value = spearmanr(inv_ppl, n_feats)

    # Approximate power for detecting correlation (using R^2 as effect size proxy)
    power = None
    try:
        # Convert to difference in proportions proxy for a rough bound
        eff = min(max(r_squared, 0.0), 1.0)
        power = estimate_power_binary(0.0, eff, n=len(estimates))
    except Exception:
        power = None

    return {
        "pearson_r": corr,
        "r_squared": r_squared,
        "spearman_r": spearman_corr,
        "spearman_p": p_value,
        "power_proxy": power,
    }


def test_low_resource_clustering(
    feature_sets: Dict[str, Set[int]],
    frequency_estimates: Dict[str, float],
) -> Dict[str, Any]:
    """Test if low-resource languages still cluster with their family."""

    # Identify low vs high resource languages by perplexity
    sorted_by_ppl = sorted(frequency_estimates.items(), key=lambda x: -x[1])
    low_resource = [l for l, _ in sorted_by_ppl[:3]]  # Top 3 by perplexity
    high_resource = [l for l, _ in sorted_by_ppl[-3:]]  # Bottom 3 by perplexity

    # Filter to available languages
    low_resource = [l for l in low_resource if l in feature_sets]
    high_resource = [l for l in high_resource if l in feature_sets]

    # Malayalam is expected to be low-resource
    ml_in_low = "ml" in low_resource

    # Check if Malayalam clusters with other Dravidian
    dravidian = [l for l in INDIC_LANGUAGES["dravidian"] if l in feature_sets]
    non_dravidian = [l for l in feature_sets if l not in dravidian]

    ml_to_dravidian = []
    ml_to_non_dravidian = []

    if "ml" in feature_sets:
        for l in dravidian:
            if l != "ml":
                ml_to_dravidian.append(jaccard_overlap(feature_sets["ml"], feature_sets[l]))
        for l in non_dravidian:
            ml_to_non_dravidian.append(jaccard_overlap(feature_sets["ml"], feature_sets[l]))

    ml_clusters_with_dravidian = (
        np.mean(ml_to_dravidian) > np.mean(ml_to_non_dravidian)
        if ml_to_dravidian and ml_to_non_dravidian else None
    )

    return {
        "low_resource_langs": low_resource,
        "high_resource_langs": high_resource,
        "malayalam_is_low_resource": ml_in_low,
        "malayalam_dravidian_overlap": np.mean(ml_to_dravidian) if ml_to_dravidian else None,
        "malayalam_non_dravidian_overlap": np.mean(ml_to_non_dravidian) if ml_to_non_dravidian else None,
        "malayalam_clusters_with_dravidian": ml_clusters_with_dravidian,
    }


def main():
    seed_everything(SEED)

    print("=" * 60)
    print("EXPERIMENT 20: Training Data Frequency Control")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in model.model_id.lower() else ""

    # Load FLORES data
    print("\nLoading FLORES data...")
    flores = load_flores(
        max_samples=N_SAMPLES_DISCOVERY,
        languages=EXTENDED_LANGUAGES,
        split="dev",
    )

    available_langs = [l for l in flores if flores[l]]
    print(f"Available languages: {available_langs}")

    # Use middle layer
    layer = TARGET_LAYERS[len(TARGET_LAYERS) // 2]
    print(f"\nAnalyzing Layer {layer}")

    results = {
        "layer": layer,
        "languages_analyzed": available_langs,
        "frequency_estimates": {},
        "feature_counts": {},
    }

    # 1. Estimate perplexity for each language
    print("\n" + "=" * 60)
    print("ESTIMATING TRAINING FREQUENCY (PERPLEXITY)")
    print("=" * 60)

    frequency_estimates = {}
    feature_sets = {}

    for lang in tqdm(available_langs, desc="Languages"):
        texts = flores[lang]
        if not texts:
            continue

        # Compute perplexity
        ppl = compute_perplexity(model, texts)
        frequency_estimates[lang] = ppl

        # Compute feature set
        feature_sets[lang] = compute_active_features(model, texts, layer)

        print(f"  {lang}: PPL={ppl:.1f}, n_features={len(feature_sets[lang])}")

        results["frequency_estimates"][lang] = ppl
        results["feature_counts"][lang] = len(feature_sets[lang])

    # Create frequency estimate objects
    estimates = []
    for lang in available_langs:
        if lang in frequency_estimates and lang in feature_sets:
            estimates.append(LanguageFrequencyEstimate(
                lang=lang,
                perplexity=frequency_estimates[lang],
                inverse_perplexity=1.0 / max(frequency_estimates[lang], 1.0),
                n_active_features=len(feature_sets[lang]),
            ))

    # 2. Correlation analysis
    print("\n" + "=" * 60)
    print("FREQUENCY vs FEATURE COUNT CORRELATION")
    print("=" * 60)

    corr_results = analyze_frequency_correlation(estimates)

    if "error" not in corr_results:
        print(f"\n  Pearson r: {corr_results['pearson_r']:.3f}")
        print(f"  R²: {corr_results['r_squared']:.3f}")
        print(f"  Spearman r: {corr_results['spearman_r']:.3f}")
        print(f"  Spearman p-value: {corr_results['spearman_p']:.4f}")

        # Interpretation
        if corr_results['r_squared'] > 0.5:
            interp = "Strong: frequency explains >50% of feature count variance"
        elif corr_results['r_squared'] > 0.25:
            interp = "Moderate: frequency explains 25-50% of variance"
        else:
            interp = "Weak: frequency explains <25% of variance"

        print(f"\n  Interpretation: {interp}")
        corr_results["interpretation"] = interp

    results["correlation_analysis"] = corr_results

    # 3. Low-resource clustering test
    print("\n" + "=" * 60)
    print("LOW-RESOURCE LANGUAGE CLUSTERING")
    print("=" * 60)

    low_resource_results = test_low_resource_clustering(feature_sets, frequency_estimates)

    print(f"\n  Low-resource languages: {low_resource_results['low_resource_langs']}")
    print(f"  High-resource languages: {low_resource_results['high_resource_langs']}")
    print(f"  Malayalam is low-resource: {low_resource_results['malayalam_is_low_resource']}")

    if low_resource_results['malayalam_dravidian_overlap'] is not None:
        print(f"  Malayalam-Dravidian overlap: {low_resource_results['malayalam_dravidian_overlap']:.3f}")
        print(f"  Malayalam-Non-Dravidian overlap: {low_resource_results['malayalam_non_dravidian_overlap']:.3f}")
        print(f"  Malayalam clusters with Dravidian: {low_resource_results['malayalam_clusters_with_dravidian']}")

    results["low_resource_analysis"] = low_resource_results

    # 4. Family vs Frequency comparison
    print("\n" + "=" * 60)
    print("FAMILY vs FREQUENCY: WHAT EXPLAINS OVERLAP?")
    print("=" * 60)

    # Compare explanatory power
    # Family-based R²: Do same-family pairs have higher overlap?
    family_overlaps = []
    non_family_overlaps = []

    all_indic = INDIC_LANGUAGES["indo_aryan"] + INDIC_LANGUAGES["dravidian"]

    for l1 in feature_sets:
        for l2 in feature_sets:
            if l1 >= l2:
                continue
            overlap = jaccard_overlap(feature_sets[l1], feature_sets[l2])

            # Same family?
            same_family = (
                (l1 in INDIC_LANGUAGES["indo_aryan"] and l2 in INDIC_LANGUAGES["indo_aryan"]) or
                (l1 in INDIC_LANGUAGES["dravidian"] and l2 in INDIC_LANGUAGES["dravidian"])
            )

            if same_family:
                family_overlaps.append(overlap)
            else:
                non_family_overlaps.append(overlap)

    if family_overlaps and non_family_overlaps:
        family_mean = np.mean(family_overlaps)
        non_family_mean = np.mean(non_family_overlaps)
        family_effect = family_mean - non_family_mean

        print(f"\n  Same-family mean overlap: {family_mean:.3f}")
        print(f"  Cross-family mean overlap: {non_family_mean:.3f}")
        print(f"  Family effect (difference): {family_effect:.3f}")

        fam_test = mann_whitney_test(family_overlaps, non_family_overlaps, alternative="greater")
        fam_power = None
        try:
            fam_power = estimate_power_independent(non_family_overlaps, family_overlaps)
        except Exception:
            fam_power = None

        results["family_vs_frequency"] = {
            "same_family_overlap": family_mean,
            "cross_family_overlap": non_family_mean,
            "family_effect": family_effect,
            "mann_whitney": fam_test.to_dict(),
            "power": fam_power,
        }

    # Multiple testing correction across the primary hypothesis tests in this experiment.
    pvals = []
    names = []
    if "error" not in corr_results and corr_results.get("spearman_p") is not None:
        pvals.append(float(corr_results["spearman_p"]))
        names.append("spearman_frequency_vs_feature_count")
    fam_block = results.get("family_vs_frequency", {}) or {}
    fam_mw = fam_block.get("mann_whitney", {}) or {}
    if fam_mw.get("p_value") is not None:
        pvals.append(float(fam_mw["p_value"]))
        names.append("family_overlap_greater_than_cross")

    if pvals:
        holm = holm_bonferroni_correction(pvals)
        results["multiple_testing"] = {
            "method": holm.get("method"),
            "n_tests": holm.get("n_comparisons"),
            "tests": [],
        }
        for i, name in enumerate(names):
            results["multiple_testing"]["tests"].append(
                {
                    "name": name,
                    "p_value": pvals[i],
                    "adjusted_p": holm["adjusted_p_values"][i],
                    "significant": holm["significant"][i],
                }
            )

    # Final interpretation
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    conclusions = []

    if corr_results.get("r_squared", 0) > 0.5:
        conclusions.append("CAUTION: Frequency may be confounding Indic cluster claims")
    else:
        conclusions.append("Frequency is NOT a major confound")

    if low_resource_results.get("malayalam_clusters_with_dravidian"):
        conclusions.append("Low-resource Malayalam still clusters with Dravidian (supports genuine linguistic structure)")
    elif low_resource_results.get("malayalam_clusters_with_dravidian") is False:
        conclusions.append("Malayalam does NOT cluster with Dravidian (frequency confound possible)")

    for c in conclusions:
        print(f"  • {c}")

    results["conclusions"] = conclusions

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp20_training_freq_control{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
