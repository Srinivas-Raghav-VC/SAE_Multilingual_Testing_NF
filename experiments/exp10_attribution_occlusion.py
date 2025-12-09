"""Experiment 10: Occlusion-Based Attribution Steering

Goal:
    Implement an attribution-inspired steering method that selects SAE
    features based on their *causal* effect on Hindi generation, using
    occlusion (clamping) rather than only activation statistics.

Idea:
    1. Select a small set of candidate features for a layer (e.g. via
       activation-diff).
    2. For each feature j:
         - Generate baseline outputs for prompts.
         - Generate outputs where feature j is ablated via SAE encode→
           zero→decode at that layer.
         - Measure change in Hindi success (script+semantic).
    3. Rank features by how much ablation *hurts* Hindi success.
    4. Build an "attribution" steering vector from top-k such features
       (decoder rows), and compare its steering performance to:
         - Dense
         - Activation-diff
         - Monolinguality

This is a simple, model-agnostic attribution method aligned with the
latent attribution philosophy: interventions, not just correlations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import List, Dict

import torch
from tqdm import tqdm

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY, STEERING_STRENGTHS, EVAL_PROMPTS
from data import load_flores
from model import GemmaWithSAE
from experiments.exp2_steering import (
    get_activation_diff_features,
    construct_sae_steering_vector,
    construct_dense_steering_vector,
)
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
    load_judge_calibration_table,
    calibrated_judge_from_results,
)


@dataclass
class FeatureAttributionScore:
    feature_idx: int
    delta_success: float


def ablate_feature_in_hidden(
    sae,
    hidden: torch.Tensor,
    feature_idx: int,
) -> torch.Tensor:
    """Ablate one SAE feature by encoding, zeroing, and decoding.

    Args:
        sae: SAE module for this layer
        hidden: (batch, seq, d_model) hidden states
        feature_idx: index of SAE feature to ablate
    """
    b, s, d = hidden.shape
    hs = hidden.view(-1, d)
    with torch.no_grad():
        z = sae.encode(hs)  # (b*s, d_sae)
        z[:, feature_idx] = 0.0
        h_abl = sae.decode(z)  # (b*s, d_model)

    # Ensure the ablated hidden states have the same dtype/device as the
    # original hidden states, otherwise later linear layers (which are
    # typically in bf16) will see a dtype mismatch.
    h_abl = h_abl.to(hidden.dtype).to(hidden.device)
    return h_abl.view(b, s, d)


def score_feature_occlusion(
    model: GemmaWithSAE,
    layer: int,
    feature_idx: int,
    prompts: List[str],
) -> float:
    """Estimate causal importance of a feature via occlusion.

    Returns:
        Mean change in script+semantic success when feature is ablated.
        Positive means ablation *hurts* Hindi success (feature supports Hindi).
    """
    sae = model.load_sae(layer)

    # Baseline outputs
    baseline_results = []
    for p in prompts:
        out = model.generate(p, max_new_tokens=64)
        baseline_results.append(
            evaluate_steering_output(
                p,
                out,
                method="baseline",
                strength=0.0,
                layer=layer,
                compute_semantics=True,
                use_llm_judge=False,
            )
        )
    base_agg = aggregate_results(baseline_results)
    base_success = base_agg.semantic_success_rate or base_agg.success_rate

    # Ablated outputs: forward hook that applies ablation at layer
    ablated_results = []

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_abl = ablate_feature_in_hidden(sae, h, feature_idx)
        return (h_abl,) + output[1:] if isinstance(output, tuple) else h_abl

    handle = model.model.model.layers[layer].register_forward_hook(hook)
    try:
        for p in prompts:
            out = model.generate(p, max_new_tokens=64)
            ablated_results.append(
                evaluate_steering_output(
                    p,
                    out,
                    method="ablated",
                    strength=0.0,
                    layer=layer,
                    compute_semantics=True,
                    use_llm_judge=False,
                )
            )
    finally:
        handle.remove()

    abl_agg = aggregate_results(ablated_results)
    abl_success = abl_agg.semantic_success_rate or abl_agg.success_rate

    # Positive delta means ablation hurt success
    return float(base_success - abl_success)


def build_attribution_steering_vector(
    model: GemmaWithSAE,
    layer: int,
    top_features: List[int],
) -> torch.Tensor:
    """Steering vector from decoder rows of top attribution features."""
    sae = model.load_sae(layer)
    directions = sae.W_dec[top_features, :]
    vec = directions.mean(dim=0)
    vec = vec / vec.norm() * (sae.cfg.d_in ** 0.5)
    return vec


def main():
    print("=" * 60)
    print("EXPERIMENT 10: Occlusion-Based Attribution Steering")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()

    # Load FLORES data
    print("\nLoading FLORES data...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    texts_en = flores.get("en", [])
    texts_hi = flores.get("hi", [])

    if not texts_en or not texts_hi:
        print("ERROR: Need en and hi data.")
        return

    # Prompts for attribution and steering evaluation
    prompts = EVAL_PROMPTS[:15]

    # For simplicity, probe a single layer from TARGET_LAYERS that is mid/late
    layer_candidates = [l for l in TARGET_LAYERS if 10 <= l <= 24]
    layer = layer_candidates[0] if layer_candidates else TARGET_LAYERS[0]
    print(f"\nUsing layer {layer} for attribution-based steering.")

    # Step 1: select candidate features by activation-diff
    print("\nSelecting candidate features (activation-diff)...")
    cand_feats = get_activation_diff_features(model, texts_en, texts_hi, layer, top_k=20)

    # Step 2: occlusion scoring
    scores: List[FeatureAttributionScore] = []
    print("\nScoring candidates via occlusion...")

    for feat in tqdm(cand_feats, desc="Occlusion scoring"):
        delta = score_feature_occlusion(model, layer, feat, prompts)
        scores.append(FeatureAttributionScore(feature_idx=feat, delta_success=delta))

    # Sort by causal importance
    scores.sort(key=lambda s: s.delta_success, reverse=True)
    top_k = 10
    top_features = [s.feature_idx for s in scores[:top_k] if s.delta_success > 0]

    print(f"\nTop attribution features (layer {layer}):")
    for s in scores[:top_k]:
        print(f"  feat {s.feature_idx}: Δsuccess={s.delta_success:.3f}")

    if not top_features:
        print("No positively-contributing features found; aborting steering comparison.")
        return

    # Step 3: build steering vectors
    print("\nBuilding steering vectors...")
    dense_vec = construct_dense_steering_vector(model, texts_en, texts_hi, layer)
    act_vec = construct_sae_steering_vector(model, layer, cand_feats[:top_k])
    attr_vec = build_attribution_steering_vector(model, layer, top_features)

    methods = {
        "dense": dense_vec,
        "activation_diff": act_vec,
        "attribution_occlusion": attr_vec,
    }

    # Step 4: steering evaluation
    print("\nEvaluating steering vectors...")
    steering_results: Dict[str, Dict] = {}
    # For calibrated judge, collect all results (Hindi target language).
    cal_table = load_judge_calibration_table()
    if not cal_table:
        print(
            "[exp10] Warning: no judge calibration statistics found. "
            "Structural metrics are still computed, but the calibrated "
            "judge summary will be skipped. Run Exp11 first if you need "
            "calibrated Gemini scores."
        )
    all_results_for_judge: List = []

    for name, vec in methods.items():
        outputs = []
        for strength in [0.5, 1.0, 2.0]:
            for p in tqdm(
                prompts,
                desc=f"{name}@{strength}",
                leave=False,
            ):
                gen = model.generate_with_steering(
                    p,
                    layer=layer,
                    steering_vector=vec,
                    strength=strength,
                    max_new_tokens=64,
                )
                outputs.append(
                    evaluate_steering_output(
                        p,
                        gen,
                        method=name,
                        strength=strength,
                        layer=layer,
                        compute_semantics=True,
                        use_llm_judge=True,
                        judge_lang="hi",
                    )
                )

        agg = aggregate_results(outputs)
        all_results_for_judge.extend(
            [r for r in outputs if r.llm_judge_raw is not None]
        )
        steering_results[name] = {
            "n_samples": agg.n_samples,
            "success_rate_script": agg.success_rate,
            "success_rate_script_semantic": agg.semantic_success_rate,
            "avg_target_script_ratio": agg.avg_target_script_ratio,
            "avg_semantic_similarity": agg.avg_semantic_similarity,
            "degradation_rate": agg.degradation_rate,
        }

    # Optional calibrated judge summary across all steering runs.
    judge_summary = None
    cj = calibrated_judge_from_results(
        all_results_for_judge, lang="hi", calibration_table=cal_table
    )
    if cj is not None:
        judge_summary = {
            "raw_accuracy": cj.raw_accuracy,
            "corrected_accuracy": cj.corrected_accuracy,
            "ci_low": cj.confidence_interval[0],
            "ci_high": cj.confidence_interval[1],
            "q0": cj.q0,
            "q1": cj.q1,
            "n_test": cj.n_test,
            "n_calib_0": cj.n_calib_0,
            "n_calib_1": cj.n_calib_1,
        }

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp10_attribution_occlusion.json"

    import json

    with open(out_path, "w") as f:
        json.dump(
            {
                "layer": layer,
                "attribution_scores": [
                    {"feature_idx": s.feature_idx, "delta_success": s.delta_success}
                    for s in scores
                ],
                "steering_results": steering_results,
                "judge_summary": judge_summary,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
