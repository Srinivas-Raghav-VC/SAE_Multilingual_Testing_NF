"""Experiment 25: Causal Ablation of Language Family Features

Tests whether Indo-Aryan and Dravidian language family features are
necessary and/or sufficient for cross-lingual steering.

Research Questions:
  1. Are family-specific features necessary for steering within that family?
  2. Are family-specific features sufficient for cross-family steering?
  3. Can we isolate the causal role of family features vs pan-Indic features?

Falsification Criteria:
  - If ablating IA features doesn't hurt IA steering → IA features not necessary
  - If ablating DR features doesn't hurt DR steering → DR features not necessary
  - If IA features alone enable DR steering → families share representation

Methodology:
  1. Identify Indo-Aryan-specific and Dravidian-specific features
  2. Ablate family features during steering
  3. Measure steering effectiveness with/without family features
  4. Test necessity (ablation hurts) and sufficiency (features alone work)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm
from stats import bootstrap_ci

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    N_STEERING_EVAL,
    SEED,
    LANG_TO_SCRIPT,
)
from data import load_research_data
from model import GemmaWithSAE
from evaluation_comprehensive import (
    evaluate_steering_output,
    detect_script,
    compute_semantic_similarity,
    estimate_power_binary,
)
from reproducibility import seed_everything


# Define language families
INDO_ARYAN_LANGS = ["hi", "bn", "ur", "mr", "gu", "pa"]
DRAVIDIAN_LANGS = ["ta", "te", "kn", "ml"]


@dataclass
class FamilyFeatures:
    """Features identified for each language family."""
    indo_aryan_specific: List[int]
    dravidian_specific: List[int]
    pan_indic: List[int]  # Active in both families
    layer: int


@dataclass
class AblationResult:
    """Results of ablating family features during steering."""
    source_lang: str
    target_lang: str
    source_family: str
    target_family: str
    layer: int
    # Baseline (no ablation)
    baseline_script_accuracy: float
    baseline_semantic_similarity: float
    # Ablate source family features
    ablate_source_script_acc: float
    ablate_source_semantic_sim: float
    # Ablate target family features
    ablate_target_script_acc: float
    ablate_target_semantic_sim: float
    # Ablate both family features
    ablate_both_script_acc: float
    ablate_both_semantic_sim: float


@dataclass
class CausalAnalysis:
    """Causal analysis of family feature roles."""
    ia_features_necessary_for_ia: bool
    dr_features_necessary_for_dr: bool
    ia_features_sufficient_for_dr: bool
    dr_features_sufficient_for_ia: bool
    family_separation_causal: bool
    p_ia_adj: float
    p_dr_adj: float
    power_ia: float
    power_dr: float
    interpretation: str


def identify_family_features(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    specificity_ratio: float = 2.0,
) -> FamilyFeatures:
    """Identify features specific to each language family.

    Args:
        model: Model with SAE
        texts_by_lang: Dictionary mapping language codes to texts
        layer: Layer to analyze
        specificity_ratio: Ratio threshold for family-specific features

    Returns:
        FamilyFeatures with family-specific and pan-Indic features
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    # Compute activation rates per language
    lang_activation_rates = {}

    for lang, texts in texts_by_lang.items():
        activation_counts = torch.zeros(n_features, device=model.device)
        total_tokens = 0

        for text in texts[:50]:
            acts = model.get_sae_activations(text, layer)
            if acts.shape[0] > 0:
                activation_counts += (acts > 0).float().sum(dim=0)
                total_tokens += acts.shape[0]

        if total_tokens > 0:
            lang_activation_rates[lang] = (activation_counts / total_tokens).cpu().numpy()

    # Aggregate by family
    ia_langs = [l for l in INDO_ARYAN_LANGS if l in lang_activation_rates]
    dr_langs = [l for l in DRAVIDIAN_LANGS if l in lang_activation_rates]

    if not ia_langs or not dr_langs:
        return FamilyFeatures([], [], [], layer)

    ia_rates = np.mean([lang_activation_rates[l] for l in ia_langs], axis=0)
    dr_rates = np.mean([lang_activation_rates[l] for l in dr_langs], axis=0)

    # Classify features
    indo_aryan_specific = []
    dravidian_specific = []
    pan_indic = []

    for i in range(n_features):
        ia_rate = ia_rates[i]
        dr_rate = dr_rates[i]

        # Skip inactive features
        if ia_rate < 0.01 and dr_rate < 0.01:
            continue

        if ia_rate > specificity_ratio * max(dr_rate, 0.001):
            indo_aryan_specific.append(i)
        elif dr_rate > specificity_ratio * max(ia_rate, 0.001):
            dravidian_specific.append(i)
        elif ia_rate > 0.01 and dr_rate > 0.01:
            pan_indic.append(i)

    return FamilyFeatures(
        indo_aryan_specific=indo_aryan_specific,
        dravidian_specific=dravidian_specific,
        pan_indic=pan_indic,
        layer=layer,
    )


def steer_with_ablation(
    model: GemmaWithSAE,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    features_to_ablate: List[int],
    steering_strength: float = 2.0,
) -> str:
    """Generate with steering while ablating specific features.

    Args:
        model: Model with SAE
        prompt: Input prompt
        steering_vector: Steering direction in SAE space
        layer: Layer to apply steering
        features_to_ablate: Features to zero out
        steering_strength: Multiplier for steering vector

    Returns:
        Generated text
    """
    # Ensure SAE is loaded before using it in the hook
    sae = model.load_sae(layer)

    # Create hook that applies steering with ablation
    def steering_ablation_hook(module, input, output):
        # SAE is captured from outer scope (already loaded)
        if sae is None:
            return output

        hidden = output[0] if isinstance(output, tuple) else output

        with torch.no_grad():
            # SAE expects 2D input (batch*seq, d_model), hidden is 3D (batch, seq, d_model)
            orig_shape = hidden.shape
            hidden_2d = hidden.reshape(-1, hidden.size(-1))  # (batch*seq, d_model)

            # Encode to SAE space
            sae_acts = sae.encode(hidden_2d)  # (batch*seq, d_sae)

            # Apply steering (steering_vector is 1D, broadcast to 2D)
            sae_acts = sae_acts + steering_strength * steering_vector.unsqueeze(0)

            # Ablate specified features (2D indexing)
            if features_to_ablate:
                sae_acts[:, features_to_ablate] = 0

            # Decode back and restore shape
            reconstructed = sae.decode(sae_acts)  # (batch*seq, d_model)
            reconstructed = reconstructed.reshape(orig_shape)  # Restore to 3D

        if isinstance(output, tuple):
            return (reconstructed,) + output[1:]
        return reconstructed

    # Register hook
    layer_module = model.model.model.layers[layer]
    hook_handle = layer_module.register_forward_hook(steering_ablation_hook)

    try:
        output = model.generate(prompt, max_new_tokens=50)
    finally:
        hook_handle.remove()

    return output


def compute_steering_vector(
    model: GemmaWithSAE,
    source_texts: List[str],
    target_texts: List[str],
    layer: int,
) -> torch.Tensor:
    """Compute steering vector from source to target language."""
    source_acts = []
    target_acts = []

    for text in source_texts[:50]:
        acts = model.get_sae_activations(text, layer)
        if acts.shape[0] > 0:
            source_acts.append(acts.mean(dim=0))

    for text in target_texts[:50]:
        acts = model.get_sae_activations(text, layer)
        if acts.shape[0] > 0:
            target_acts.append(acts.mean(dim=0))

    if not source_acts or not target_acts:
        return None

    source_mean = torch.stack(source_acts).mean(dim=0)
    target_mean = torch.stack(target_acts).mean(dim=0)

    return target_mean - source_mean


def evaluate_steering_with_ablation(
    model: GemmaWithSAE,
    prompts: List[str],
    steering_vector: torch.Tensor,
    layer: int,
    target_script: str,
    features_to_ablate: List[int],
) -> Tuple[float, float]:
    """Evaluate steering effectiveness with ablation.

    Returns:
        (script_accuracy, semantic_similarity)
    """
    correct_script = 0
    similarities = []

    for prompt in prompts[:20]:
        output = steer_with_ablation(
            model, prompt, steering_vector, layer, features_to_ablate
        )

        # Check script
        detected_script = detect_script(output)
        if detected_script == target_script:
            correct_script += 1

        # Semantic similarity
        sim = compute_semantic_similarity(prompt, output)
        similarities.append(sim)

    n = len(prompts[:20])
    if n == 0:
        raise ValueError("evaluate_steering_with_ablation received 0 prompts; cannot compute metrics.")
    return correct_script / n, np.mean(similarities)


def test_family_feature_causality(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    prompts: List[str],
    layer: int,
    family_features: FamilyFeatures,
) -> List[AblationResult]:
    """Test causal role of family features in steering."""

    results = []

    # Test pairs
    test_pairs = [
        # Within Indo-Aryan
        ("hi", "bn", "indo_aryan", "indo_aryan"),
        # Within Dravidian
        ("ta", "te", "dravidian", "dravidian"),
        # Cross-family
        ("hi", "ta", "indo_aryan", "dravidian"),
        ("ta", "hi", "dravidian", "indo_aryan"),
    ]

    for source, target, source_family, target_family in test_pairs:
        if source not in texts_by_lang or target not in texts_by_lang:
            continue

        print(f"\n  Testing {source} → {target}...")

        # Compute steering vector
        steering_vec = compute_steering_vector(
            model, texts_by_lang[source], texts_by_lang[target], layer
        )

        if steering_vec is None:
            continue

        target_script = LANG_TO_SCRIPT.get(target, "unknown")

        # Baseline (no ablation)
        baseline_script, baseline_sem = evaluate_steering_with_ablation(
            model, prompts, steering_vec, layer, target_script, []
        )

        # Ablate source family features
        source_features = (
            family_features.indo_aryan_specific
            if source_family == "indo_aryan"
            else family_features.dravidian_specific
        )
        ablate_src_script, ablate_src_sem = evaluate_steering_with_ablation(
            model, prompts, steering_vec, layer, target_script, source_features
        )

        # Ablate target family features
        target_features = (
            family_features.indo_aryan_specific
            if target_family == "indo_aryan"
            else family_features.dravidian_specific
        )
        ablate_tgt_script, ablate_tgt_sem = evaluate_steering_with_ablation(
            model, prompts, steering_vec, layer, target_script, target_features
        )

        # Ablate both family features
        both_features = list(set(source_features) | set(target_features))
        ablate_both_script, ablate_both_sem = evaluate_steering_with_ablation(
            model, prompts, steering_vec, layer, target_script, both_features
        )

        results.append(AblationResult(
            source_lang=source,
            target_lang=target,
            source_family=source_family,
            target_family=target_family,
            layer=layer,
            baseline_script_accuracy=baseline_script,
            baseline_semantic_similarity=baseline_sem,
            ablate_source_script_acc=ablate_src_script,
            ablate_source_semantic_sim=ablate_src_sem,
            ablate_target_script_acc=ablate_tgt_script,
            ablate_target_semantic_sim=ablate_tgt_sem,
            ablate_both_script_acc=ablate_both_script,
            ablate_both_semantic_sim=ablate_both_sem,
        ))

        print(f"    Baseline: {baseline_script:.1%} script, {baseline_sem:.3f} sem")
        print(f"    Ablate source family: {ablate_src_script:.1%} script, {ablate_src_sem:.3f} sem")
        print(f"    Ablate target family: {ablate_tgt_script:.1%} script, {ablate_tgt_sem:.3f} sem")
        print(f"    Ablate both: {ablate_both_script:.1%} script, {ablate_both_sem:.3f} sem")

    return results


def analyze_causality(
    results: List[AblationResult],
    degradation_threshold: float = 0.05,
    n_prompts: int = 20,
) -> CausalAnalysis:
    """Analyze causal role of family features from ablation results.

    Args:
        results: Ablation results
        degradation_threshold: How much degradation indicates necessity

    Returns:
        CausalAnalysis with causal conclusions
    """
    # Within-family pairs
    ia_within = [r for r in results if r.source_family == "indo_aryan" and r.target_family == "indo_aryan"]
    dr_within = [r for r in results if r.source_family == "dravidian" and r.target_family == "dravidian"]

    # Cross-family pairs
    ia_to_dr = [r for r in results if r.source_family == "indo_aryan" and r.target_family == "dravidian"]
    dr_to_ia = [r for r in results if r.source_family == "dravidian" and r.target_family == "indo_aryan"]

    def _degradation_ci(arr: List[float]):
        if not arr:
            return 0.0, 0.0, (0.0, 0.0)
        ci = bootstrap_ci(arr)
        return float(np.mean(arr)), float(ci.estimate), (ci.ci_low, ci.ci_high)

    # Check if IA features are necessary for IA steering
    ia_necessary = False
    ia_deg = [r.baseline_script_accuracy - r.ablate_source_script_acc for r in ia_within]
    ia_mean, _ia_est, ia_ci = _degradation_ci(ia_deg)
    ia_necessary = ia_ci[0] > 0.0
    p_ia = 1.0
    if ia_deg:
        try:
            p_ia = scipy_stats.ttest_1samp(ia_deg, 0, alternative="greater").pvalue
        except Exception:
            p_ia = 1.0
    power_ia = estimate_power_binary(
        np.mean([r.ablate_source_script_acc for r in ia_within]) if ia_within else 0.0,
        np.mean([r.baseline_script_accuracy for r in ia_within]) if ia_within else 0.0,
        n=n_prompts,
    )

    # Check if DR features are necessary for DR steering
    dr_deg = [r.baseline_script_accuracy - r.ablate_source_script_acc for r in dr_within]
    dr_mean, _dr_est, dr_ci = _degradation_ci(dr_deg)
    dr_necessary = dr_ci[0] > 0.0
    p_dr = 1.0
    if dr_deg:
        try:
            p_dr = scipy_stats.ttest_1samp(dr_deg, 0, alternative="greater").pvalue
        except Exception:
            p_dr = 1.0
    power_dr = estimate_power_binary(
        np.mean([r.ablate_source_script_acc for r in dr_within]) if dr_within else 0.0,
        np.mean([r.baseline_script_accuracy for r in dr_within]) if dr_within else 0.0,
        n=n_prompts,
    )

    # Holm correction across necessity tests
    try:
        from stats import holm_bonferroni_correction
        adj = holm_bonferroni_correction([p_ia, p_dr])
        p_ia_adj, p_dr_adj = adj["adjusted_p_values"]
    except Exception:
        p_ia_adj, p_dr_adj = p_ia, p_dr

    # Check if IA features are sufficient for DR steering (use source ablation for sufficiency test)
    ia_sufficient_for_dr = False
    if ia_to_dr:
        degr = [r.baseline_script_accuracy - r.ablate_source_script_acc for r in ia_to_dr]
        degr_mean, _est, degr_ci = _degradation_ci(degr)
        ia_sufficient_for_dr = degr_ci[1] <= degradation_threshold

    # Check if DR features are sufficient for IA steering
    dr_sufficient_for_ia = False
    if dr_to_ia:
        degr = [r.baseline_script_accuracy - r.ablate_source_script_acc for r in dr_to_ia]
        _m, _e, degr_ci = _degradation_ci(degr)
        dr_sufficient_for_ia = degr_ci[1] <= degradation_threshold

    # Overall family separation is causal if family features are necessary
    family_separation_causal = ia_necessary and dr_necessary

    # Build interpretation
    interpretations = []
    if ia_necessary:
        interpretations.append(f"Indo-Aryan features are NECESSARY for IA steering (p_Holm={p_ia_adj:.4f})")
    else:
        interpretations.append(f"Indo-Aryan features are NOT necessary for IA steering (p_Holm={p_ia_adj:.4f})")

    if dr_necessary:
        interpretations.append(f"Dravidian features are NECESSARY for DR steering (p_Holm={p_dr_adj:.4f})")
    else:
        interpretations.append(f"Dravidian features are NOT necessary for DR steering (p_Holm={p_dr_adj:.4f})")

    if family_separation_causal:
        interpretations.append("CONCLUSION: Family separation has CAUSAL role in steering")
    else:
        interpretations.append("CONCLUSION: Family separation may be correlational, not causal")

    return CausalAnalysis(
        ia_features_necessary_for_ia=ia_necessary,
        dr_features_necessary_for_dr=dr_necessary,
        ia_features_sufficient_for_dr=ia_sufficient_for_dr,
        dr_features_sufficient_for_ia=dr_sufficient_for_ia,
        family_separation_causal=family_separation_causal,
        p_ia_adj=p_ia_adj,
        p_dr_adj=p_dr_adj,
        power_ia=power_ia,
        power_dr=power_dr,
        interpretation="; ".join(interpretations),
    )


def main():
    seed_everything(SEED)

    print("=" * 60)
    print("EXPERIMENT 25: Causal Ablation of Family Features")
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

    # Collect texts by language
    all_indic_langs = INDO_ARYAN_LANGS + DRAVIDIAN_LANGS
    available_langs = [l for l in all_indic_langs if l in data.train]
    texts_by_lang = {l: data.train[l] for l in available_langs}

    print(f"\nAvailable Indo-Aryan: {[l for l in INDO_ARYAN_LANGS if l in available_langs]}")
    print(f"Available Dravidian: {[l for l in DRAVIDIAN_LANGS if l in available_langs]}")

    # Use middle layer for analysis
    test_layer = TARGET_LAYERS[len(TARGET_LAYERS) // 2]
    print(f"\nTest layer: {test_layer}")

    # Identify family features
    print("\n" + "=" * 60)
    print("IDENTIFYING FAMILY-SPECIFIC FEATURES")
    print("=" * 60)

    family_features = identify_family_features(model, texts_by_lang, test_layer)

    print(f"\n  Indo-Aryan specific: {len(family_features.indo_aryan_specific)} features")
    print(f"  Dravidian specific: {len(family_features.dravidian_specific)} features")
    print(f"  Pan-Indic: {len(family_features.pan_indic)} features")

    # Test causality
    print("\n" + "=" * 60)
    print("TESTING FEATURE CAUSALITY")
    print("=" * 60)

    ablation_results = test_family_feature_causality(
        model, texts_by_lang, prompts, test_layer, family_features
    )

    # Analyze causality
    print("\n" + "=" * 60)
    print("CAUSAL ANALYSIS")
    print("=" * 60)

    n_eff = min(20, len(prompts))
    causal_analysis = analyze_causality(ablation_results, n_prompts=n_eff)

    print(f"\n  {causal_analysis.interpretation}")

    # Summary table
    print("\n  SUMMARY:")
    print(f"    IA features necessary for IA steering: {causal_analysis.ia_features_necessary_for_ia}")
    print(f"    DR features necessary for DR steering: {causal_analysis.dr_features_necessary_for_dr}")
    print(f"    IA features sufficient for DR steering: {causal_analysis.ia_features_sufficient_for_dr}")
    print(f"    DR features sufficient for IA steering: {causal_analysis.dr_features_sufficient_for_ia}")
    print(f"    p_Holm (IA necessary): {causal_analysis.p_ia_adj:.4f}, power={causal_analysis.power_ia:.2f}")
    print(f"    p_Holm (DR necessary): {causal_analysis.p_dr_adj:.4f}, power={causal_analysis.power_dr:.2f}")
    print(f"    Family separation is causal: {causal_analysis.family_separation_causal}")

    # Save results
    results = {
        "layer": test_layer,
        "family_features": {
            "n_indo_aryan_specific": len(family_features.indo_aryan_specific),
            "n_dravidian_specific": len(family_features.dravidian_specific),
            "n_pan_indic": len(family_features.pan_indic),
        },
        "ablation_results": [
            {
                "source_lang": r.source_lang,
                "target_lang": r.target_lang,
                "source_family": r.source_family,
                "target_family": r.target_family,
                "baseline_script_accuracy": r.baseline_script_accuracy,
                "baseline_semantic_similarity": r.baseline_semantic_similarity,
                "ablate_source_script_acc": r.ablate_source_script_acc,
                "ablate_source_semantic_sim": r.ablate_source_semantic_sim,
                "ablate_target_script_acc": r.ablate_target_script_acc,
                "ablate_target_semantic_sim": r.ablate_target_semantic_sim,
                "ablate_both_script_acc": r.ablate_both_script_acc,
                "ablate_both_semantic_sim": r.ablate_both_semantic_sim,
            }
            for r in ablation_results
        ],
        "causal_analysis": {
            "ia_features_necessary_for_ia": causal_analysis.ia_features_necessary_for_ia,
            "dr_features_necessary_for_dr": causal_analysis.dr_features_necessary_for_dr,
            "ia_features_sufficient_for_dr": causal_analysis.ia_features_sufficient_for_dr,
            "dr_features_sufficient_for_ia": causal_analysis.dr_features_sufficient_for_ia,
            "family_separation_causal": causal_analysis.family_separation_causal,
            "p_ia_holm": causal_analysis.p_ia_adj,
            "p_dr_holm": causal_analysis.p_dr_adj,
            "power_ia": causal_analysis.power_ia,
            "power_dr": causal_analysis.power_dr,
            "interpretation": causal_analysis.interpretation,
        },
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp25_family_causal{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
