"""Experiment 8: Scaling to Gemma 2 9B and Low-Resource Languages

Goals:
    1. Compare Indic language representation and steering between:
         - Gemma 2 2B + Gemma Scope 2B SAEs
         - Gemma 2 9B + Gemma Scope 9B SAEs (if available)
    2. Evaluate steering effects on low-resource languages:
         - Low-resource Indic (e.g., as, or)
         - Low-resource non-Indic (e.g., ar, vi)

This is intentionally lightweight:
    - Uses a small subset of layers and prompts
    - Reuses feature-discovery + steering comparison ideas
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List
from dataclasses import dataclass

import torch

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    STEERING_STRENGTHS,
    EVAL_PROMPTS,
    MODEL_ID,
    MODEL_ID_9B,
    SAE_RELEASE_2B,
    SAE_RELEASE_9B,
)
from data import load_flores
from model import GemmaWithSAE
from experiments.exp2_steering import (
    construct_sae_steering_vector,
    construct_dense_steering_vector,
)
from evaluation_comprehensive import evaluate_steering_output, aggregate_results


@dataclass
class ModelConfig:
    name: str
    model_id: str
    sae_release: str


def compute_language_feature_counts(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    threshold: float = 0.01,
) -> Dict[str, int]:
    """Rough per-language active-feature counts for a layer."""
    counts = {}
    for lang, texts in texts_by_lang.items():
        if not texts:
            continue
        sae = model.load_sae(layer)
        n_features = sae.cfg.d_sae
        acc = torch.zeros(n_features, device=model.device)
        total_tokens = 0
        for t in texts[:200]:
            acts = model.get_sae_activations(t, layer)
            acc += (acts > 0).float().sum(dim=0)
            total_tokens += acts.shape[0]
        rates = acc / max(total_tokens, 1)
        counts[lang] = int((rates > threshold).sum().item())
    return counts


def simple_hi_steering_eval(
    model: GemmaWithSAE,
    texts_en: List[str],
    texts_hi: List[str],
    layer: int,
    prompts: List[str],
) -> Dict[str, Dict]:
    """Evaluate dense vs SAE steering for EN→HI at one layer."""
    from experiments.exp2_steering import get_activation_diff_features

    results: Dict[str, Dict] = {}

    # Dense steering vector
    dense_vec = construct_dense_steering_vector(model, texts_en, texts_hi, layer)

    # SAE activation-diff features
    act_feats = get_activation_diff_features(model, texts_en, texts_hi, layer, top_k=25)
    sae_vec = construct_sae_steering_vector(model, layer, act_feats)

    for name, vec in [("dense", dense_vec), ("sae_activation_diff", sae_vec)]:
        outputs = []
        for strength in [1.0, 2.0]:
            for p in prompts:
                gen = model.generate_with_steering(
                    p, layer=layer, steering_vector=vec, strength=strength, max_new_tokens=64
                )
                outputs.append(
                    evaluate_steering_output(
                        p,
                        gen,
                        method=name,
                        strength=strength,
                        layer=layer,
                        compute_semantics=False,  # semantic model is optional
                    )
                )
        agg = aggregate_results(outputs)
        results[name] = {
            "success_rate": agg.success_rate,
            "avg_target_script_ratio": agg.avg_target_script_ratio,
            "degradation_rate": agg.degradation_rate,
        }

    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 8: Scaling to 9B and Low-Resource Languages")
    print("=" * 60)

    # Two model variants: 2B (default) and 9B (if available)
    model_variants = [
        ModelConfig(name="gemma-2-2b", model_id=MODEL_ID, sae_release=SAE_RELEASE_2B),
        ModelConfig(name="gemma-2-9b", model_id=MODEL_ID_9B, sae_release=SAE_RELEASE_9B),
    ]

    # Load FLORES data once
    print("\nLoading FLORES data...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)

    # Indic + low-resource-ish languages
    indic_langs = ["hi", "ur", "bn", "as", "or"]
    non_indic_low = ["ar", "vi"]  # Semitic + SE Asian
    langs_of_interest = [l for l in indic_langs + non_indic_low if l in flores]

    texts_by_lang = {l: flores[l] for l in langs_of_interest}
    texts_en = flores.get("en", [])
    texts_hi = flores.get("hi", [])

    if not texts_en or not texts_hi:
        print("ERROR: Need English and Hindi data for steering comparison.")
        return

    prompts = EVAL_PROMPTS[:10]

    # Use a few representative layers
    layers_to_test = [l for l in TARGET_LAYERS if l in {5, 13, 20, 24}]
    if not layers_to_test:
        layers_to_test = [13]

    all_results = {}

    for cfg in model_variants:
        print(f"\n=== Model: {cfg.name} ===")
        try:
            model = GemmaWithSAE(model_id=cfg.model_id, sae_release=cfg.sae_release)
            model.load_model()
        except Exception as e:
            print(f"  Skipping {cfg.name}: could not load model/SAE ({e})")
            continue

        model_results = {"feature_counts": {}, "steering": {}}

        for layer in layers_to_test:
            print(f"\n  Layer {layer}: feature coverage")
            counts = compute_language_feature_counts(model, texts_by_lang, layer)
            model_results["feature_counts"][str(layer)] = counts

            print(f"  Layer {layer}: EN→HI steering")
            steering = simple_hi_steering_eval(model, texts_en, texts_hi, layer, prompts)
            model_results["steering"][str(layer)] = steering

        all_results[cfg.name] = model_results

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    import json

    with open(out_dir / "exp8_scaling_9b_low_resource.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {out_dir / 'exp8_scaling_9b_low_resource.json'}")


if __name__ == "__main__":
    main()
