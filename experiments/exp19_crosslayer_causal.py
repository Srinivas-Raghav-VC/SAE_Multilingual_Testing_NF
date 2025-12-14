"""Experiment 19: Cross-Layer Causal Profiles

Extends the causal ablation analysis from Exp10/13 to ALL target layers.

Research Questions:
  1. Does the layer with most Indic features also have the most causal impact?
  2. Is causal importance distributed across layers or concentrated?
  3. Do Indo-Aryan and Dravidian show different causal layer profiles?

Methodology:
  - For each layer in TARGET_LAYERS:
    * Identify top-k Indic features via activation-diff
    * Ablate these features and measure script/semantic degradation
    * Compute causal importance = |baseline_success - ablated_success|
  - Compare causal profile to correlational profile (feature counts from Exp1)

Falsification Criteria:
  - If layers with most features != layers with most causal impact,
    feature counting is misleading for claims about "where language lives"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import torch
import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    LANG_TO_SCRIPT,
    INDIC_LANGUAGES,
    ALL_INDIC,
    SEED,
)
from data import load_research_data
from model import GemmaWithSAE
from steering_utils import get_activation_diff_features
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
)
from reproducibility import seed_everything
from stats import bootstrap_ci
from evaluation_comprehensive import estimate_power_binary


@dataclass
class LayerCausalProfile:
    """Causal analysis results for a single layer."""
    layer: int
    target_lang: str
    n_features_ablated: int
    baseline_success: float
    ablated_success: float
    random_ablated_success: float
    causal_importance: float  # |baseline - ablated|
    random_causal_importance: float
    baseline_script_ratio: float
    ablated_script_ratio: float
    random_script_ratio: float
    script_drop: float
    random_script_drop: float
    p_value_ablate: float
    p_value_rand: float
    adjusted_p_ablate: float = 1.0
    adjusted_p_rand: float = 1.0
    baseline_sensitivity: Any = None
    ablated_sensitivity: Any = None
    random_sensitivity: Any = None


def ablate_features_in_hidden(
    sae,
    hidden: torch.Tensor,
    feature_indices: List[int],
) -> torch.Tensor:
    """Ablate multiple SAE features by encoding, zeroing, and decoding."""
    b, s, d = hidden.shape
    hs = hidden.reshape(-1, d)
    with torch.no_grad():
        z = sae.encode(hs)
        for idx in feature_indices:
            z[:, idx] = 0.0
        h_abl = sae.decode(z)
    h_abl = h_abl.to(hidden.dtype).to(hidden.device)
    return h_abl.reshape(b, s, d)


def compute_layer_causal_profile(
    model: GemmaWithSAE,
    layer: int,
    target_lang: str,
    train_data: Dict[str, List[str]],
    prompts: List[str],
    n_features: int = 25,
) -> LayerCausalProfile:
    """Compute causal importance of Indic features at a specific layer."""

    target_script = LANG_TO_SCRIPT.get(target_lang, "devanagari")

    # Get top features for this layer (EN -> target_lang)
    features = get_activation_diff_features(
        model,
        texts_src=train_data["en"],
        texts_tgt=train_data[target_lang],
        layer=layer,
        top_k=n_features,
    )

    sae = model.load_sae(layer)
    total_feats = int(getattr(sae.cfg, "d_sae", 0))
    rng = np.random.default_rng(SEED + layer)
    random_features: List[int] = []
    if total_feats > 0 and len(features) > 0:
        random_features = rng.choice(total_feats, size=len(features), replace=False).tolist()

    # Cap SAE cache during this sweep to prevent VRAM blow-up across layers.
    # If a caller set a larger cache globally, this local override keeps the run safe.
    os.environ.setdefault("SAE_CACHE_SIZE", "1")

    # Compute baseline outputs
    baseline_results = []
    per_prompt_success_base: List[int] = []
    for p in tqdm(prompts, desc=f"Baseline L{layer}", leave=False):
        out = model.generate(p, max_new_tokens=64)
        res = evaluate_steering_output(
            p, out, method="baseline", strength=0.0, layer=layer,
            compute_semantics=True, target_script=target_script,
        )
        baseline_results.append(res)
        per_prompt_success_base.append(1 if res.overall_success else 0)

    base_agg = aggregate_results(baseline_results, target_script=target_script)
    baseline_success = (
        base_agg.semantic_success_rate
        if base_agg.semantic_success_rate is not None
        else base_agg.success_rate
    )
    baseline_script_ratio = base_agg.avg_target_script_ratio

    # Compute ablated outputs
    def ablation_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_abl = ablate_features_in_hidden(sae, h, features)
        return (h_abl,) + output[1:] if isinstance(output, tuple) else h_abl

    handle = model.model.model.layers[layer].register_forward_hook(ablation_hook)

    ablated_results = []
    per_prompt_success_abl: List[int] = []
    try:
        for p in tqdm(prompts, desc=f"Ablated L{layer}", leave=False):
            out = model.generate(p, max_new_tokens=64)
            res = evaluate_steering_output(
                p, out, method="ablated", strength=0.0, layer=layer,
                compute_semantics=True, target_script=target_script,
            )
            ablated_results.append(res)
            per_prompt_success_abl.append(1 if res.overall_success else 0)
    finally:
        handle.remove()

    abl_agg = aggregate_results(ablated_results, target_script=target_script)
    ablated_success = (
        abl_agg.semantic_success_rate
        if abl_agg.semantic_success_rate is not None
        else abl_agg.success_rate
    )
    ablated_script_ratio = abl_agg.avg_target_script_ratio

    # Paired test baseline vs ablated (overall success as binary per prompt)
    from stats import wilcoxon_test
    p_value_ablate = 1.0
    try:
        if len(per_prompt_success_base) == len(per_prompt_success_abl) and per_prompt_success_base:
            p_value_ablate = wilcoxon_test(per_prompt_success_base, per_prompt_success_abl).p_value
    except Exception:
        p_value_ablate = 1.0

    # Random-control ablation (size-matched)
    rand_success = baseline_success
    rand_script_ratio = baseline_script_ratio
    p_value_rand = 1.0
    if random_features:
        def rand_hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_abl = ablate_features_in_hidden(sae, h, random_features)
            return (h_abl,) + output[1:] if isinstance(output, tuple) else h_abl

        handle_rand = model.model.model.layers[layer].register_forward_hook(rand_hook)
        rand_results = []
        per_prompt_success_rand: List[int] = []
        try:
            for p in tqdm(prompts, desc=f"Rand Ablation L{layer}", leave=False):
                out = model.generate(p, max_new_tokens=64)
                res = evaluate_steering_output(
                    p, out, method="rand_ablation", strength=0.0, layer=layer,
                    compute_semantics=True, target_script=target_script,
                )
                rand_results.append(res)
                per_prompt_success_rand.append(1 if res.overall_success else 0)
        finally:
            handle_rand.remove()

        rand_agg = aggregate_results(rand_results, target_script=target_script)
        rand_success = (
            rand_agg.semantic_success_rate
            if rand_agg.semantic_success_rate is not None
            else rand_agg.success_rate
        )
        rand_script_ratio = rand_agg.avg_target_script_ratio
        try:
            if len(per_prompt_success_base) == len(per_prompt_success_rand) and per_prompt_success_base:
                p_value_rand = wilcoxon_test(per_prompt_success_base, per_prompt_success_rand).p_value
        except Exception:
            p_value_rand = 1.0

    try:
        del sae
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return LayerCausalProfile(
        layer=layer,
        target_lang=target_lang,
        n_features_ablated=len(features),
        baseline_success=baseline_success,
        ablated_success=ablated_success,
        random_ablated_success=rand_success,
        causal_importance=abs(baseline_success - ablated_success),
        random_causal_importance=abs(baseline_success - rand_success),
        baseline_script_ratio=baseline_script_ratio,
        ablated_script_ratio=ablated_script_ratio,
        random_script_ratio=rand_script_ratio,
        script_drop=baseline_script_ratio - ablated_script_ratio,
        random_script_drop=baseline_script_ratio - rand_script_ratio,
        p_value_ablate=p_value_ablate,
        p_value_rand=p_value_rand,
        baseline_sensitivity=base_agg.sensitivity,
        ablated_sensitivity=abl_agg.sensitivity,
        random_sensitivity=rand_agg.sensitivity if random_features else None,
    )


def correlate_causal_with_feature_counts(
    causal_profiles: List[LayerCausalProfile],
    feature_counts_by_layer: Dict[int, int],
) -> Tuple[float, str]:
    """Compute correlation between causal importance and feature counts."""
    layers = [p.layer for p in causal_profiles]
    causal = [p.causal_importance for p in causal_profiles]
    counts = [feature_counts_by_layer.get(l, 0) for l in layers]

    if len(layers) < 3:
        return 0.0, "insufficient_data"

    # Spearman rank correlation
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(causal, counts)

    interpretation = "strong_positive" if corr > 0.7 else \
                     "moderate_positive" if corr > 0.4 else \
                     "weak_positive" if corr > 0.1 else \
                     "no_correlation" if corr > -0.1 else \
                     "weak_negative" if corr > -0.4 else \
                     "moderate_negative" if corr > -0.7 else "strong_negative"

    return corr, interpretation


def main():
    print("=" * 60)
    print("EXPERIMENT 19: Cross-Layer Causal Profiles")
    print("=" * 60)

    seed_everything(SEED)

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

    prompts = data.steering_prompts[:50]  # Use subset for efficiency
    print(f"Using {len(prompts)} prompts for causal analysis")

    # Target languages: one Indo-Aryan, one Dravidian
    target_langs = ["hi", "ta"]  # Hindi (Indo-Aryan) and Tamil (Dravidian)

    results = {
        "layers": TARGET_LAYERS,
        "causal_profiles": {},
        "family_comparison": {},
    }

    for target_lang in target_langs:
        if target_lang not in data.train:
            print(f"Skipping {target_lang}: no training data")
            continue

        print(f"\n=== Target: {target_lang} ({LANG_TO_SCRIPT.get(target_lang, 'unknown')}) ===")

        profiles = []
        for layer in TARGET_LAYERS:
            print(f"\nProcessing Layer {layer}...")
            profile = compute_layer_causal_profile(
                model, layer, target_lang, data.train, prompts
            )
            profiles.append(profile)

            print(f"  Baseline success: {profile.baseline_success:.1%}")
            print(f"  Ablated success:  {profile.ablated_success:.1%}")
            print(f"  Causal importance: {profile.causal_importance:.3f}")
            print(f"  Script drop: {profile.script_drop:.1%}")

        # Find peak causal layer
        if not profiles:
            print(f"  WARNING: No profiles computed for {target_lang}")
            continue

        # Holm correction across layers for ablation and random-control p-values
        from stats import holm_bonferroni_correction
        p_ablate = [p.p_value_ablate for p in profiles]
        p_rand = [p.p_value_rand for p in profiles]
        if p_ablate:
            adj = holm_bonferroni_correction(p_ablate)
            for prof, adj_p in zip(profiles, adj["adjusted_p_values"]):
                prof.adjusted_p_ablate = adj_p
        if p_rand:
            adj = holm_bonferroni_correction(p_rand)
            for prof, adj_p in zip(profiles, adj["adjusted_p_values"]):
                prof.adjusted_p_rand = adj_p

        # Power estimates for primary endpoint (success rate difference)
        for prof in profiles:
            prof.power_ablate = estimate_power_binary(
                prof.baseline_success,
                prof.ablated_success,
                n=len(prompts),
            )
            prof.power_rand = estimate_power_binary(
                prof.baseline_success,
                prof.random_ablated_success,
                n=len(prompts),
            )
        peak_profile = max(profiles, key=lambda p: p.causal_importance)
        print(f"\n  Peak causal layer: {peak_profile.layer} "
              f"(importance={peak_profile.causal_importance:.3f})")

        # Store results
        results["causal_profiles"][target_lang] = [
            {
                "layer": p.layer,
                "baseline_success": p.baseline_success,
                "ablated_success": p.ablated_success,
                "random_ablated_success": p.random_ablated_success,
                "causal_importance": p.causal_importance,
                "random_causal_importance": p.random_causal_importance,
                "baseline_script_ratio": p.baseline_script_ratio,
                "ablated_script_ratio": p.ablated_script_ratio,
                "random_script_ratio": p.random_script_ratio,
                "script_drop": p.script_drop,
                "random_script_drop": p.random_script_drop,
                "p_value_ablate": p.p_value_ablate,
                "p_value_rand": p.p_value_rand,
                "p_value_ablate_holm": p.adjusted_p_ablate,
                "p_value_rand_holm": p.adjusted_p_rand,
                "power_ablate": getattr(p, "power_ablate", None),
                "power_rand": getattr(p, "power_rand", None),
                "baseline_sensitivity": p.baseline_sensitivity,
                "ablated_sensitivity": p.ablated_sensitivity,
                "random_sensitivity": p.random_sensitivity,
            }
            for p in profiles
        ]

        results["family_comparison"][target_lang] = {
            "peak_causal_layer": peak_profile.layer,
            "peak_causal_importance": peak_profile.causal_importance,
            "max_script_drop_layer": max(profiles, key=lambda p: p.script_drop).layer,
        }

    # Cross-family comparison
    print("\n" + "=" * 60)
    print("CROSS-FAMILY CAUSAL COMPARISON")
    print("=" * 60)

    if "hi" in results["family_comparison"] and "ta" in results["family_comparison"]:
        hi_peak = results["family_comparison"]["hi"]["peak_causal_layer"]
        ta_peak = results["family_comparison"]["ta"]["peak_causal_layer"]

        same_peak = hi_peak == ta_peak
        results["cross_family"] = {
            "indo_aryan_peak_layer": hi_peak,
            "dravidian_peak_layer": ta_peak,
            "same_peak_layer": same_peak,
            "interpretation": (
                "Unified Indic causal profile" if same_peak
                else "Family-specific causal profiles"
            ),
        }

        print(f"  Indo-Aryan (Hindi) peak: Layer {hi_peak}")
        print(f"  Dravidian (Tamil) peak: Layer {ta_peak}")
        print(f"  Same peak? {same_peak}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for lang, profiles_data in results["causal_profiles"].items():
        importances = [p["causal_importance"] for p in profiles_data]
        mean_imp = np.mean(importances)
        std_imp = np.std(importances)
        print(f"\n{lang.upper()}:")
        print(f"  Mean causal importance: {mean_imp:.3f} ± {std_imp:.3f}")
        print(f"  Layers by importance: {sorted(profiles_data, key=lambda p: -p['causal_importance'])[:3]}")

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp19_crosslayer_causal{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
