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

import os
import torch
from tqdm import tqdm
import random
import numpy as np

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY, STEERING_STRENGTHS, EVAL_PROMPTS
from data import load_flores, load_steering_prompts
from model import GemmaWithSAE
from steering_utils import (
    get_activation_diff_features,
    get_random_features,
    construct_sae_steering_vector,
    construct_dense_steering_vector,
)
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
    get_semantic_truncation_stats,
    load_judge_calibration_table,
    calibrated_judge_from_results,
)
from reproducibility import seed_everything


@dataclass
class FeatureAttributionScore:
    feature_idx: int
    delta_success: float


@dataclass
class LogitAttributionScore:
    feature_idx: int
    delta_script_prob_mass: float


_SCRIPT_TOKEN_MASK_CACHE: Dict[str, torch.Tensor] = {}


def _char_in_ranges(char: str, ranges) -> bool:
    code = ord(char)
    if isinstance(ranges, tuple) and len(ranges) == 2:
        return ranges[0] <= code <= ranges[1]
    if isinstance(ranges, list):
        for start, end in ranges:
            if start <= code <= end:
                return True
        return False
    return False


def _get_script_token_mask(model: GemmaWithSAE, target_script: str = "devanagari") -> torch.Tensor:
    """Return a boolean mask over vocab tokens that contain target-script chars."""
    from config import SCRIPT_RANGES

    key = f"{getattr(model, 'model_id', '')}:{target_script}"
    if key in _SCRIPT_TOKEN_MASK_CACHE:
        return _SCRIPT_TOKEN_MASK_CACHE[key]

    ranges = SCRIPT_RANGES.get(target_script)
    if ranges is None:
        raise ValueError(f"Unknown target_script '{target_script}'. Add to config.SCRIPT_RANGES.")

    vocab = model.tokenizer.get_vocab()
    max_idx = max(vocab.values()) if vocab else -1
    cfg_vocab_size = getattr(getattr(getattr(model, "model", None), "config", None), "vocab_size", None) or 0
    # Some tokenizers use non-contiguous IDs; ensure the mask covers the max ID.
    vocab_size = max(int(cfg_vocab_size), int(max_idx + 1), len(vocab))
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tok, idx in vocab.items():
        if any(_char_in_ranges(c, ranges) for c in tok):
            if 0 <= idx < vocab_size:
                mask[idx] = True

    _SCRIPT_TOKEN_MASK_CACHE[key] = mask.to(model.device)
    return _SCRIPT_TOKEN_MASK_CACHE[key]


def _next_token_script_prob_mass(
    model: GemmaWithSAE,
    prompt: str,
    token_mask: torch.Tensor,
    layer: int,
    sae=None,
    feature_idx: int | None = None,
) -> float:
    """Compute P(next token is target-script) with optional feature ablation at `layer`."""
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)

    handle = None
    if feature_idx is not None:
        if sae is None:
            sae = model.load_sae(layer)

        def hook(module, inputs_, output):
            h = output[0] if isinstance(output, tuple) else output
            h_abl = ablate_feature_in_hidden(sae, h, feature_idx)
            return (h_abl,) + output[1:] if isinstance(output, tuple) else h_abl

        handle = model.model.model.layers[layer].register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = model.model(**inputs)
            logits = out.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            mask = token_mask
            if mask.numel() != probs.numel():
                # Align sizes defensively (do not crash due to tokenizer/model ID quirks).
                if mask.numel() < probs.numel():
                    tmp = torch.zeros_like(probs, dtype=torch.bool)
                    tmp[: mask.numel()] = mask.to(tmp.device)
                    mask = tmp
                else:
                    mask = mask[: probs.numel()]
            return float(probs[mask].sum().item())
    finally:
        if handle is not None:
            handle.remove()


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
    hs = hidden.reshape(-1, d)
    with torch.no_grad():
        z = sae.encode(hs)  # (b*s, d_sae)
        z[:, feature_idx] = 0.0
        h_abl = sae.decode(z)  # (b*s, d_model)

    # Ensure the ablated hidden states have the same dtype/device as the
    # original hidden states, otherwise later linear layers (which are
    # typically in bf16) will see a dtype mismatch.
    h_abl = h_abl.to(hidden.dtype).to(hidden.device)
    return h_abl.reshape(b, s, d)


def score_feature_occlusion(
    model: GemmaWithSAE,
    layer: int,
    feature_idx: int,
    prompts: List[str],
    baseline_success: float,
) -> float:
    """Estimate causal importance of a feature via occlusion.

    Returns:
        Mean change in script+semantic success when feature is ablated.
        Positive means ablation *hurts* Hindi success (feature supports Hindi).
    """
    sae = model.load_sae(layer)

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
    abl_success = (
        abl_agg.semantic_success_rate
        if abl_agg.semantic_success_rate is not None
        else abl_agg.success_rate
    )

    # Positive delta means ablation hurt success
    return float(baseline_success - abl_success)


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

    from config import SEED
    seed_everything(SEED)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in str(getattr(model, "model_id", "")).lower() else ""

    # Load FLORES data
    print("\nLoading FLORES data...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    texts_en = flores.get("en", [])
    texts_hi = flores.get("hi", [])

    if not texts_en or not texts_hi:
        print("ERROR: Need en and hi data.")
        return

    # Prompts for attribution (selection) vs evaluation (held-out).
    # Use a deterministic split to prevent selection/evaluation leakage.
    all_prompts = load_steering_prompts(min_prompts=200)
    n_occl = max(1, int(os.environ.get("N_ATTRIBUTION_PROMPTS", "60")))
    n_eval = max(1, int(os.environ.get("N_ATTRIBUTION_EVAL_PROMPTS", "100")))
    if len(all_prompts) < n_occl + n_eval:
        # Deterministically extend by cycling to avoid randomness.
        extra = (n_occl + n_eval) - len(all_prompts)
        all_prompts = (all_prompts * ((extra // len(all_prompts)) + 2))[: n_occl + n_eval]
    prompts = all_prompts[:n_occl]
    eval_prompts = all_prompts[n_occl : n_occl + n_eval]

    # ------------------------------------------------------------------
    # Semantic reference mode (steering eval only)
    # ------------------------------------------------------------------
    # Default: prompt-faithfulness via LaBSE(prompt, output).
    # Optional: SEMANTIC_REFERENCE=baseline to measure baseline-preservation
    # via LaBSE(baseline, steered) on the steering evaluation prompts.
    semantic_reference_mode = os.environ.get("SEMANTIC_REFERENCE", "prompt").strip().lower()
    baseline_refs_eval: Dict[str, str] = {}
    if semantic_reference_mode in ("baseline", "base"):
        print("[exp10] SEMANTIC_REFERENCE=baseline: computing baseline references for eval prompts.")
        for p in tqdm(eval_prompts, desc="Baseline refs (eval)", leave=False):
            baseline_refs_eval[p] = model.generate(p, max_new_tokens=64)
        semantic_reference_mode = "baseline"
    else:
        semantic_reference_mode = "prompt"

    # For simplicity, probe a single layer from TARGET_LAYERS that is mid/late
    layer_candidates = [l for l in TARGET_LAYERS if 10 <= l <= 24]
    layer = layer_candidates[0] if layer_candidates else TARGET_LAYERS[0]
    print(f"\nUsing layer {layer} for attribution-based steering.")

    # Step 1: select candidate features by activation-diff
    print("\nSelecting candidate features (activation-diff)...")
    cand_feats = get_activation_diff_features(model, texts_en, texts_hi, layer, top_k=20)

    # Matched random-feature control: occlude random features from the same
    # layer to show that causal drops are not a generic "any feature matters"
    # artifact.
    sae = model.load_sae(layer)
    n_random = int(os.environ.get("N_ATTRIBUTION_RANDOM_CONTROLS", "10"))
    random_feats = [f for f in get_random_features(sae, top_k=len(cand_feats)) if f not in cand_feats]
    random_feats = random_feats[:n_random]

    # ------------------------------------------------------------------
    # Baseline caching (critical for compute + determinism)
    # ------------------------------------------------------------------
    print("\nComputing baseline outputs once (for attribution scoring)...")
    baseline_results = []
    for p in tqdm(prompts, desc="Baseline generation", leave=False):
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
    baseline_success = (
        base_agg.semantic_success_rate
        if base_agg.semantic_success_rate is not None
        else base_agg.success_rate
    )
    print(f"[exp10] Baseline success (script+semantic if available): {baseline_success:.3f}")

    # Step 2a: causal attribution via generation-based occlusion
    scores: List[FeatureAttributionScore] = []
    print("\nScoring candidates via occlusion (Δ success)...")

    for feat in tqdm(cand_feats, desc="Occlusion scoring", leave=False):
        delta = score_feature_occlusion(
            model, layer, feat, prompts, baseline_success=baseline_success
        )
        scores.append(FeatureAttributionScore(feature_idx=feat, delta_success=delta))

    random_control_scores: List[FeatureAttributionScore] = []
    if random_feats:
        print("\nScoring random control features via occlusion...")
        for feat in tqdm(random_feats, desc="Random occlusion control", leave=False):
            delta = score_feature_occlusion(
                model, layer, feat, prompts, baseline_success=baseline_success
            )
            random_control_scores.append(
                FeatureAttributionScore(feature_idx=feat, delta_success=delta)
            )

    # Step 2b: causal attribution via logit-level occlusion (cheaper signal)
    print("\nScoring candidates via logit attribution (Δ P(next token in target script))...")
    token_mask = _get_script_token_mask(model, target_script="devanagari")
    base_masses = [
        _next_token_script_prob_mass(model, p, token_mask, layer=layer, feature_idx=None)
        for p in tqdm(prompts, desc="Baseline logits", leave=False)
    ]
    base_mass = float(sum(base_masses) / max(1, len(base_masses)))
    logit_scores: List[LogitAttributionScore] = []
    sae = model.load_sae(layer)
    for feat in tqdm(cand_feats, desc="Logit attribution", leave=False):
        masses = [
            _next_token_script_prob_mass(
                model,
                p,
                token_mask,
                layer=layer,
                sae=sae,
                feature_idx=feat,
            )
            for p in prompts
        ]
        abl_mass = float(sum(masses) / max(1, len(masses)))
        logit_scores.append(
            LogitAttributionScore(feature_idx=feat, delta_script_prob_mass=base_mass - abl_mass)
        )

    logit_random_control_scores: List[LogitAttributionScore] = []
    if random_feats:
        print("\nScoring random control features via logit attribution...")
        for feat in tqdm(random_feats, desc="Random logit attribution control", leave=False):
            masses = [
                _next_token_script_prob_mass(
                    model,
                    p,
                    token_mask,
                    layer=layer,
                    sae=sae,
                    feature_idx=feat,
                )
                for p in prompts
            ]
            abl_mass = float(sum(masses) / max(1, len(masses)))
            logit_random_control_scores.append(
                LogitAttributionScore(feature_idx=feat, delta_script_prob_mass=base_mass - abl_mass)
            )

    # Sort by causal importance
    scores.sort(key=lambda s: s.delta_success, reverse=True)
    top_k = 10
    top_features = [s.feature_idx for s in scores[:top_k] if s.delta_success > 0]

    print(f"\nTop attribution features (layer {layer}):")
    for s in scores[:top_k]:
        print(f"  feat {s.feature_idx}: Δsuccess={s.delta_success:.3f}")

    logit_scores.sort(key=lambda s: s.delta_script_prob_mass, reverse=True)
    top_features_logit = [
        s.feature_idx for s in logit_scores[:top_k] if s.delta_script_prob_mass > 0
    ]
    print(f"\nTop logit-attribution features (layer {layer}):")
    for s in logit_scores[:top_k]:
        print(f"  feat {s.feature_idx}: ΔP(script)={s.delta_script_prob_mass:.4f}")

    if not top_features and not top_features_logit:
        print("No positively-contributing features found; aborting steering comparison.")
        return

    # Step 3: build steering vectors
    print("\nBuilding steering vectors...")
    dense_vec = construct_dense_steering_vector(model, texts_en, texts_hi, layer)
    act_vec = construct_sae_steering_vector(model, layer, cand_feats[:top_k])
    methods: Dict[str, torch.Tensor] = {
        "dense": dense_vec,
        "activation_diff": act_vec,
    }
    if top_features:
        attr_vec = build_attribution_steering_vector(model, layer, top_features)
        methods["attribution_occlusion"] = attr_vec
    if top_features_logit:
        logit_vec = build_attribution_steering_vector(model, layer, top_features_logit)
        methods["attribution_logit"] = logit_vec
    # Random negative-control steering vector (same k).
    rand_feats_vec = [
        f for f in get_random_features(sae, top_k=top_k + len(cand_feats)) if f not in cand_feats
    ][:top_k]
    methods["random"] = construct_sae_steering_vector(model, layer, rand_feats_vec)

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
                eval_prompts,
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
                ref = baseline_refs_eval.get(p) if baseline_refs_eval else None
                outputs.append(
                    evaluate_steering_output(
                        p,
                        gen,
                        method=name,
                        strength=strength,
                        layer=layer,
                        semantic_reference=ref,
                        compute_semantics=True,
                        use_llm_judge=True,
                        judge_lang="hi",
                    )
                )

        agg = aggregate_results(outputs, target_script="devanagari")
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
            "sensitivity": agg.sensitivity,
            "separate_metrics": agg.separate_metrics.to_dict() if agg.separate_metrics else None,
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
    out_path = out_dir / f"exp10_attribution_occlusion{suffix}.json"

    import json

    with open(out_path, "w") as f:
        rand_mean = None
        if random_control_scores:
            rand_mean = float(
                sum(s.delta_success for s in random_control_scores) / len(random_control_scores)
            )
        rand_logit_mean = None
        if logit_random_control_scores:
            rand_logit_mean = float(
                sum(s.delta_script_prob_mass for s in logit_random_control_scores) / len(logit_random_control_scores)
            )
        json.dump(
            {
                "layer": layer,
                "semantic_reference": semantic_reference_mode,
                "semantic_truncation_stats": get_semantic_truncation_stats(),
                "baseline_success": baseline_success,
                "attribution_scores": [
                    {"feature_idx": s.feature_idx, "delta_success": s.delta_success}
                    for s in scores
                ],
                "logit_attribution_scores": [
                    {
                        "feature_idx": s.feature_idx,
                        "delta_script_prob_mass": s.delta_script_prob_mass,
                    }
                    for s in logit_scores
                ],
                "logit_random_control_scores": [
                    {
                        "feature_idx": s.feature_idx,
                        "delta_script_prob_mass": s.delta_script_prob_mass,
                    }
                    for s in logit_random_control_scores
                ],
                "random_control_scores": [
                    {"feature_idx": s.feature_idx, "delta_success": s.delta_success}
                    for s in random_control_scores
                ],
                "random_control_mean_delta": rand_mean,
                "logit_random_control_mean_delta": rand_logit_mean,
                "steering_results": steering_results,
                "judge_summary": judge_summary,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
