"""Experiment 7: Causal Probing of Candidate SAE Features

Goal:
    Estimate the *causal* effect of individual SAE features on Hindi
    generation, instead of relying only on activation correlations.

Method:
    1. Select candidate features at a given layer using:
        - Activation difference (HI - EN)
        - Monolinguality
    2. For each feature j:
        - Construct a steering vector from decoder row j
        - Run generation with:
            - Positive steering (+alpha * v_j)
            - Negative steering (-alpha * v_j)
        - Evaluate changes in:
            - Hindi script ratio
            - Degradation metrics
            - Optional semantic similarity and LLM-as-judge scores

The output is a per-feature table of estimated causal effect sizes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
from tqdm import tqdm

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY, N_SAMPLES_EVAL, STEERING_STRENGTHS, EVAL_PROMPTS, MIN_PROMPTS_CAUSAL_PROBING
from data import load_research_data
from model import GemmaWithSAE
from experiments.exp1_feature_discovery import compute_activation_rates, compute_monolinguality
from evaluation_comprehensive import evaluate_steering_output, aggregate_results


@dataclass
class FeatureEffect:
    layer: int
    feature_idx: int
    method: str
    strength: float
    delta_script: float
    delta_semantic: float
    delta_degradation: float


def select_candidates(
    model: GemmaWithSAE,
    texts_en: List[str],
    texts_hi: List[str],
    layer: int,
    top_k: int = 10,
) -> Dict[str, List[int]]:
    """Select candidate features via activation-diff and monolinguality."""
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    # Activation-diff
    en_acts = []
    for text in texts_en[:200]:
        acts = model.get_sae_activations(text, layer)
        en_acts.append(acts.mean(dim=0))
    en_mean = torch.stack(en_acts).mean(dim=0)

    hi_acts = []
    for text in texts_hi[:200]:
        acts = model.get_sae_activations(text, layer)
        hi_acts.append(acts.mean(dim=0))
    hi_mean = torch.stack(hi_acts).mean(dim=0)

    diff = hi_mean - en_mean
    _, top_diff = diff.topk(top_k)

    # Monolinguality (requires EN + HI rates)
    flores = {"en": texts_en[:200], "hi": texts_hi[:200]}
    rates_by_lang = {
        lang: compute_activation_rates(model, sents, layer)
        for lang, sents in flores.items()
    }
    mono = compute_monolinguality(rates_by_lang, "hi")
    _, top_mono = mono.topk(top_k)

    return {
        "activation_diff": top_diff.tolist(),
        "monolinguality": top_mono.tolist(),
    }


def build_feature_vector(model: GemmaWithSAE, layer: int, feature_idx: int) -> torch.Tensor:
    """Construct steering vector from a single SAE decoder row."""
    sae = model.load_sae(layer)
    # Each row of W_dec is a direction in hidden space
    vec = sae.W_dec[feature_idx, :]  # (d_model,)
    # Normalize to reasonable magnitude
    vec = vec / vec.norm() * (sae.cfg.d_in ** 0.5)
    return vec


def probe_feature(
    model: GemmaWithSAE,
    feature_idx: int,
    layer: int,
    prompts: List[str],
    strengths: List[float],
) -> List[FeatureEffect]:
    """Probe a single feature with +alpha and -alpha steering."""
    base_outputs = []
    for p in prompts:
        out = model.generate(p, max_new_tokens=64)
        base_outputs.append(
            evaluate_steering_output(p, out, method="baseline", layer=layer, strength=0.0)
        )
    base_agg = aggregate_results(base_outputs)

    base_script = base_agg.avg_target_script_ratio
    base_sem = base_agg.avg_semantic_similarity if base_agg.avg_semantic_similarity is not None else 0.0
    base_deg = base_agg.degradation_rate

    vec = build_feature_vector(model, layer, feature_idx)
    effects: List[FeatureEffect] = []

    for direction, tag in [(+1.0, "pos"), (-1.0, "neg")]:
        for strength in strengths:
            outputs = []
            for p in tqdm(prompts, desc=f"feat {feature_idx} {tag}@{strength}", leave=False):
                gen = model.generate_with_steering(
                    p,
                    layer=layer,
                    steering_vector=direction * vec,
                    strength=strength,
                    max_new_tokens=64,
                )
                outputs.append(
                    evaluate_steering_output(
                        p,
                        gen,
                        method=f"feature_{tag}",
                        strength=strength,
                        layer=layer,
                        compute_semantics=True,
                    )
                )

            agg = aggregate_results(outputs)
            script = agg.avg_target_script_ratio
            sem = agg.avg_semantic_similarity if agg.avg_semantic_similarity is not None else 0.0
            deg = agg.degradation_rate

            effects.append(
                FeatureEffect(
                    layer=layer,
                    feature_idx=feature_idx,
                    method=tag,
                    strength=strength,
                    delta_script=float(script - base_script),
                    delta_semantic=float(sem - base_sem),
                    delta_degradation=float(deg - base_deg),
                )
            )

    return effects


def main():
    print("=" * 60)
    print("EXPERIMENT 7: Causal Probing of SAE Features")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()

    # Load data using the unified research loader so that:
    # - candidate selection uses training texts (train split),
    # - probing prompts come from held-out steering prompts
    #   (EVAL_PROMPTS + FLORES EN devtest).
    print("\nLoading research data for Exp7 (causal probing)...")
    data_split = load_research_data()
    train_data = data_split.train
    texts_en = train_data.get("en", [])
    texts_hi = train_data.get("hi", [])
    prompts_all = data_split.steering_prompts

    if not texts_en or not texts_hi:
        print("ERROR: Need training data for en and hi.")
        return

    # Use a reasonably sized prompt set so that estimated causal effects
    # are not dominated by noise. We draw prompts from the steering
    # prompts (held-out FLORES EN + EVAL_PROMPTS), keeping them
    # distinct from the training texts used for candidate selection.
    #
    # MIDDLE-GROUND SETTINGS for paper quality:
    # - 50 prompts: decent statistical power for detecting 10-15% effects
    # - 5 features/method × 2 methods = 10 features per layer
    # - 3 strengths × 2 directions = 6 conditions per feature
    # Total: 4 layers × 10 features × 6 conditions × 50 prompts = 12,000 generations
    # Estimated time: ~6 hours
    num_prompts = min(50, len(prompts_all))
    prompts = prompts_all[:num_prompts]
    print(f"[exp7] Using {num_prompts} prompts for causal probing (middle-ground for paper)")

    # Use a small subset of layers for cost
    probe_layers = [l for l in TARGET_LAYERS if l in {5, 13, 20, 24}]
    if not probe_layers:
        probe_layers = [13]

    all_effects: List[FeatureEffect] = []

    for layer in probe_layers:
        # Clear CUDA memory between layers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nSelecting candidates at layer {layer}...")
        # Use top_k=5 to keep runtime manageable while still covering feature space
        # Total: 5 features × 2 methods = 10 features per layer
        candidates = select_candidates(model, texts_en, texts_hi, layer, top_k=5)

        for method_name, feat_list in candidates.items():
            for feat in feat_list:
                print(f"\nProbing feature {feat} (method={method_name}, layer={layer})...")
                effects = probe_feature(
                    model,
                    feature_idx=feat,
                    layer=layer,
                    prompts=prompts,
                    strengths=[0.5, 1.0, 2.0],  # Full range for paper quality
                )
                all_effects.extend(effects)

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    import json

    serialized = [
        {
            "layer": e.layer,
            "feature_idx": e.feature_idx,
            "method": e.method,
            "strength": e.strength,
            "delta_script": e.delta_script,
            "delta_semantic": e.delta_semantic,
            "delta_degradation": e.delta_degradation,
        }
        for e in all_effects
    ]

    with open(out_dir / "exp7_causal_feature_probing.json", "w") as f:
        json.dump(serialized, f, indent=2)

    print(f"\n✓ Results saved to {out_dir / 'exp7_causal_feature_probing.json'}")


if __name__ == "__main__":
    main()
