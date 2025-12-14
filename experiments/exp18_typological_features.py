"""Experiment 18: Typological Feature Analysis

Tests whether SAE clustering is due to typological features (shared due to
areal contact) rather than genetic family relationships.

Research Questions:
  1. Do languages with retroflex consonants cluster regardless of family?
  2. Do SOV languages (including non-Indic like Japanese) cluster together?
  3. Does typology or genetic family explain more variance in feature overlap?

Falsification Criteria:
  - If Japanese clusters with Hindi (both SOV) but German doesn't, typology > family
  - If retroflex languages form a sub-cluster within the broader space, areal features
    may be driving the "Indic cluster" rather than genetic relationship

Key Typological Features Tested:
  - Retroflex consonants (shared Indo-Aryan + Dravidian areal feature)
  - SOV word order (all Indic + Japanese, Korean, Turkish)
  - Agglutination (Dravidian > Indo-Aryan)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    TYPOLOGICAL_FEATURES,
    TYPOLOGICAL_CONTROLS,
    INDIC_LANGUAGES,
    ALL_INDIC,
    EXTENDED_LANGUAGES,
    SEED,
)
from data import load_flores
from model import GemmaWithSAE
from evaluation_comprehensive import jaccard_overlap
from reproducibility import seed_everything


@dataclass
class TypologicalClusterResult:
    """Results for typological feature clustering analysis."""
    feature_name: str
    languages_with_feature: List[str]
    languages_without_feature: List[str]
    within_group_overlap: float
    between_group_overlap: float
    separation_ratio: float  # within / between


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

    for text in texts[:100]:  # Subset for efficiency
        acts = model.get_sae_activations(text, layer)
        activation_counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]

    if total_tokens == 0:
        return set()

    rates = activation_counts / total_tokens
    active_mask = rates > threshold
    return set(int(i) for i in active_mask.nonzero(as_tuple=False).flatten().tolist())


def compute_group_overlap(
    feature_sets: Dict[str, Set[int]],
    group1: List[str],
    group2: List[str],
) -> float:
    """Compute mean pairwise Jaccard overlap between two groups."""
    overlaps = []
    for lang1 in group1:
        if lang1 not in feature_sets:
            continue
        for lang2 in group2:
            if lang2 not in feature_sets:
                continue
            if lang1 == lang2:
                continue
            overlap = jaccard_overlap(feature_sets[lang1], feature_sets[lang2])
            overlaps.append(overlap)

    return np.mean(overlaps) if overlaps else 0.0


def analyze_typological_feature(
    feature_sets: Dict[str, Set[int]],
    feature_name: str,
    langs_with: List[str],
    langs_without: List[str],
) -> TypologicalClusterResult:
    """Analyze clustering for a specific typological feature."""

    # Filter to languages we have data for
    langs_with = [l for l in langs_with if l in feature_sets]
    langs_without = [l for l in langs_without if l in feature_sets]

    if len(langs_with) < 2 or len(langs_without) < 2:
        return TypologicalClusterResult(
            feature_name=feature_name,
            languages_with_feature=langs_with,
            languages_without_feature=langs_without,
            within_group_overlap=0.0,
            between_group_overlap=0.0,
            separation_ratio=0.0,
        )

    # Within-group overlap (languages with the feature)
    within_overlap = compute_group_overlap(feature_sets, langs_with, langs_with)

    # Between-group overlap
    between_overlap = compute_group_overlap(feature_sets, langs_with, langs_without)

    # Separation ratio (higher = better clustering)
    separation = within_overlap / max(between_overlap, 0.01)

    return TypologicalClusterResult(
        feature_name=feature_name,
        languages_with_feature=langs_with,
        languages_without_feature=langs_without,
        within_group_overlap=within_overlap,
        between_group_overlap=between_overlap,
        separation_ratio=separation,
    )


def regression_analysis(
    feature_sets: Dict[str, Set[int]],
    typological_features: Dict[str, List[str]],
    family_groups: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Analyze whether typology or family explains more variance in overlap.

    Uses a simple approach: compute R² for each predictor.
    """
    langs = list(feature_sets.keys())

    # Build overlap matrix
    overlaps = {}
    for i, l1 in enumerate(langs):
        for l2 in langs[i+1:]:
            overlaps[(l1, l2)] = jaccard_overlap(feature_sets[l1], feature_sets[l2])

    if not overlaps:
        return {"error": "insufficient_data"}

    # Predictor 1: Same family?
    family_same = []
    for (l1, l2), overlap in overlaps.items():
        same = False
        for family, members in family_groups.items():
            if l1 in members and l2 in members:
                same = True
                break
        family_same.append((1 if same else 0, overlap))

    # Predictor 2: Share retroflex?
    retroflex = typological_features.get("retroflex", [])
    retroflex_same = []
    for (l1, l2), overlap in overlaps.items():
        same = (l1 in retroflex) == (l2 in retroflex)
        retroflex_same.append((1 if same else 0, overlap))

    # Simple correlation-based R²
    def r_squared(pairs):
        if not pairs:
            return 0.0
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        if x.std() == 0 or y.std() == 0:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return corr ** 2

    return {
        "family_r_squared": r_squared(family_same),
        "retroflex_r_squared": r_squared(retroflex_same),
        "n_pairs": len(overlaps),
    }


def main():
    seed_everything(SEED)

    print("=" * 60)
    print("EXPERIMENT 18: Typological Feature Analysis")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in model.model_id.lower() else ""

    # Load FLORES data for all available languages
    print("\nLoading FLORES data...")

    # Combine core + extended + typological control languages
    all_lang_codes = {}
    all_lang_codes.update({k: v for k, v in EXTENDED_LANGUAGES.items()})
    all_lang_codes.update(TYPOLOGICAL_CONTROLS)

    flores = load_flores(
        max_samples=N_SAMPLES_DISCOVERY,
        languages=all_lang_codes,
        split="dev",
    )

    available_langs = list(flores.keys())
    print(f"Available languages: {available_langs}")

    # Use middle layer for analysis
    layer = TARGET_LAYERS[len(TARGET_LAYERS) // 2]
    print(f"\nAnalyzing Layer {layer}...")

    # Compute feature sets for all languages
    print("\nComputing feature sets...")
    feature_sets = {}
    for lang, texts in tqdm(flores.items(), desc="Languages"):
        if not texts:
            continue
        feature_sets[lang] = compute_activation_features(model, texts, layer)
        print(f"  {lang}: {len(feature_sets[lang])} active features")

    results = {
        "layer": layer,
        "languages_analyzed": list(feature_sets.keys()),
        "typological_analyses": {},
        "family_analyses": {},
    }

    # Analyze typological features
    print("\n" + "=" * 60)
    print("TYPOLOGICAL FEATURE CLUSTERING")
    print("=" * 60)

    # 1. Retroflex analysis
    print("\n1. RETROFLEX CONSONANTS")
    retroflex_result = analyze_typological_feature(
        feature_sets,
        "retroflex",
        TYPOLOGICAL_FEATURES["retroflex"],
        TYPOLOGICAL_FEATURES["no_retroflex"],
    )
    print(f"   With retroflex: {retroflex_result.languages_with_feature}")
    print(f"   Without retroflex: {retroflex_result.languages_without_feature}")
    print(f"   Within-group overlap: {retroflex_result.within_group_overlap:.3f}")
    print(f"   Between-group overlap: {retroflex_result.between_group_overlap:.3f}")
    print(f"   Separation ratio: {retroflex_result.separation_ratio:.2f}")

    results["typological_analyses"]["retroflex"] = {
        "langs_with": retroflex_result.languages_with_feature,
        "langs_without": retroflex_result.languages_without_feature,
        "within_overlap": retroflex_result.within_group_overlap,
        "between_overlap": retroflex_result.between_group_overlap,
        "separation": retroflex_result.separation_ratio,
    }

    # 2. SOV word order
    print("\n2. SOV WORD ORDER")
    sov_langs = TYPOLOGICAL_FEATURES.get("sov", [])
    # Add typological controls that are SOV
    sov_langs_extended = sov_langs + ["ja", "ko", "tr"]
    non_sov = TYPOLOGICAL_FEATURES.get("svo", []) + TYPOLOGICAL_FEATURES.get("vso", [])

    sov_result = analyze_typological_feature(
        feature_sets,
        "sov_word_order",
        sov_langs_extended,
        non_sov,
    )
    print(f"   SOV languages: {sov_result.languages_with_feature}")
    print(f"   Non-SOV languages: {sov_result.languages_without_feature}")
    print(f"   Within-group overlap: {sov_result.within_group_overlap:.3f}")
    print(f"   Between-group overlap: {sov_result.between_group_overlap:.3f}")
    print(f"   Separation ratio: {sov_result.separation_ratio:.2f}")

    results["typological_analyses"]["sov_order"] = {
        "langs_with": sov_result.languages_with_feature,
        "langs_without": sov_result.languages_without_feature,
        "within_overlap": sov_result.within_group_overlap,
        "between_overlap": sov_result.between_group_overlap,
        "separation": sov_result.separation_ratio,
    }

    # 3. Family-based analysis
    print("\n" + "=" * 60)
    print("GENETIC FAMILY CLUSTERING")
    print("=" * 60)

    # Indo-Aryan vs Dravidian
    print("\n3. INDO-ARYAN vs DRAVIDIAN")
    indo_aryan = [l for l in INDIC_LANGUAGES["indo_aryan"] if l in feature_sets]
    dravidian = [l for l in INDIC_LANGUAGES["dravidian"] if l in feature_sets]

    ia_ia_overlap = compute_group_overlap(feature_sets, indo_aryan, indo_aryan)
    dr_dr_overlap = compute_group_overlap(feature_sets, dravidian, dravidian)
    ia_dr_overlap = compute_group_overlap(feature_sets, indo_aryan, dravidian)

    print(f"   Indo-Aryan languages: {indo_aryan}")
    print(f"   Dravidian languages: {dravidian}")
    print(f"   Indo-Aryan internal overlap: {ia_ia_overlap:.3f}")
    print(f"   Dravidian internal overlap: {dr_dr_overlap:.3f}")
    print(f"   Cross-family overlap: {ia_dr_overlap:.3f}")

    # Are they distinct sub-clusters?
    family_separation = (ia_ia_overlap + dr_dr_overlap) / 2 / max(ia_dr_overlap, 0.01)
    print(f"   Family separation ratio: {family_separation:.2f}")

    distinct_families = ia_ia_overlap > ia_dr_overlap and dr_dr_overlap > ia_dr_overlap
    print(f"   Distinct sub-clusters? {distinct_families}")

    results["family_analyses"]["indic_families"] = {
        "indo_aryan_langs": indo_aryan,
        "dravidian_langs": dravidian,
        "indo_aryan_overlap": ia_ia_overlap,
        "dravidian_overlap": dr_dr_overlap,
        "cross_family_overlap": ia_dr_overlap,
        "separation_ratio": family_separation,
        "distinct_subclusters": distinct_families,
    }

    # 4. Regression: Family vs Typology
    print("\n" + "=" * 60)
    print("VARIANCE EXPLAINED: FAMILY vs TYPOLOGY")
    print("=" * 60)

    family_groups = {
        "indo_aryan": INDIC_LANGUAGES["indo_aryan"],
        "dravidian": INDIC_LANGUAGES["dravidian"],
        "germanic": ["en", "de"],
        "semitic": ["ar"],
    }

    reg_results = regression_analysis(feature_sets, TYPOLOGICAL_FEATURES, family_groups)

    if "error" not in reg_results:
        print(f"\n   Family R²: {reg_results['family_r_squared']:.3f}")
        print(f"   Retroflex R²: {reg_results['retroflex_r_squared']:.3f}")
        print(f"   N pairs: {reg_results['n_pairs']}")

        if reg_results['family_r_squared'] > reg_results['retroflex_r_squared']:
            winner = "FAMILY > TYPOLOGY"
        else:
            winner = "TYPOLOGY > FAMILY"
        print(f"\n   Winner: {winner}")

        results["variance_analysis"] = reg_results
        results["variance_analysis"]["interpretation"] = winner

    # Summary
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    interpretations = []

    if retroflex_result.separation_ratio > 1.5:
        interpretations.append("Retroflex consonants create meaningful clustering")

    if sov_result.separation_ratio > 1.5:
        interpretations.append("SOV word order creates meaningful clustering")

    if distinct_families:
        interpretations.append("Indo-Aryan and Dravidian are distinct sub-clusters")
    else:
        interpretations.append("Indo-Aryan and Dravidian form a unified Indic cluster")

    for interp in interpretations:
        print(f"  • {interp}")

    results["interpretations"] = interpretations

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp18_typological_features{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
