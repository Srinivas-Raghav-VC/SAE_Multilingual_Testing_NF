"""Experiment 9: Layer-wise Steering Sweep with LLM Judge

Research questions:
  1. Which layers are most effective for steering EN→target language outputs?
  2. Does one steering method (dense vs SAE-based) dominate across layers?
  3. Is the best steering layer similar across Indic vs non-Indic languages?

Approach:
  - Use Samanantar (+ FLORES fallback) for robust steering vector estimation.
  - For each target language and each layer in TARGET_LAYERS:
      * Build steering vectors for multiple methods.
      * Generate steered outputs on shared prompts.
      * Evaluate with script, repetition, semantic similarity, and optional
        LLM-as-judge (Gemini) scores.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json
import os
import random

from tqdm import tqdm
import torch
import numpy as np

from config import (
    TARGET_LAYERS,
    STEERING_STRENGTHS,
    NUM_FEATURES,
    N_SAMPLES_DISCOVERY,
    N_STEERING_EVAL,
    LANG_TO_SCRIPT,
)
from data import load_research_data
from model import GemmaWithSAE
from steering_utils import (
    get_activation_diff_features,
    get_monolinguality_features,
    get_random_features,
    construct_sae_steering_vector,
    construct_dense_steering_vector,
)
from stats import compute_all_pairwise_comparisons, holm_bonferroni_correction
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
    is_gemini_available,
    load_judge_calibration_table,
    calibrated_judge_from_results,
    get_semantic_truncation_stats,
)
from reproducibility import seed_everything


@dataclass
class SteeringConfig:
    target_lang: str   # e.g. "hi"
    source_lang: str   # usually "en"
    methods: List[str] # ["dense", "activation_diff", "monolinguality", "random"]


def build_steering_vector(
    method: str,
    model: GemmaWithSAE,
    train_data: Dict[str, List[str]],
    layer: int,
    target_lang: str,
    source_lang: str = "en",
    num_features: int = NUM_FEATURES,
) -> torch.Tensor:
    """Construct steering vector for a given method, layer, and language."""
    if source_lang not in train_data or target_lang not in train_data:
        raise ValueError(f"Missing data for {source_lang} or {target_lang}")

    texts_src = train_data[source_lang]
    texts_tgt = train_data[target_lang]

    if method == "dense":
        return construct_dense_steering_vector(model, texts_src, texts_tgt, layer)

    sae = model.load_sae(layer)

    if method == "activation_diff":
        feats = get_activation_diff_features(model, texts_src, texts_tgt, layer, num_features)
        return construct_sae_steering_vector(model, layer, feats)

    if method == "monolinguality":
        # Use all languages present in train_data for monolinguality computation
        feats = get_monolinguality_features(model, train_data, layer, target_lang, num_features)
        return construct_sae_steering_vector(model, layer, feats)

    if method == "random":
        feats = get_random_features(sae, num_features)
        return construct_sae_steering_vector(model, layer, feats)

    raise ValueError(f"Unknown steering method: {method}")


def run_layer_sweep():
    print("=" * 60)
    print("EXPERIMENT 9: Layer-wise Steering Sweep with LLM Judge")
    print("=" * 60)

    # Reproducibility
    from config import SEED
    seed_everything(SEED)

    # Check Gemini judge once. If unavailable (no key / hard failure),
    # we proceed with structural metrics only.
    gemini_ok = is_gemini_available()
    if not gemini_ok:
        print(
            "[exp9] Gemini judge unavailable; continuing Exp9 with structural metrics only. "
            "Set GEMINI_API_KEY and rerun to record judge scores."
        )

    # Optional: subsample judge calls to respect RPM/RPD.
    # Set GEMINI_SAMPLE_RATE in [0,1]. Example: 0.2 => judge every ~5 prompts.
    sample_rate = float(os.environ.get("GEMINI_SAMPLE_RATE", "1.0"))
    if not gemini_ok:
        sample_rate = 0.0
    if sample_rate <= 0:
        judge_stride = None
    elif sample_rate >= 1:
        judge_stride = 1
    else:
        judge_stride = max(1, int(round(1.0 / sample_rate)))
    if judge_stride is not None:
        print(f"[exp9] Gemini judge enabled on ~1/{judge_stride} prompts (sample_rate={sample_rate}).")

    # Load research data (Samanantar + FLORES + QA + prompts)
    data_split = load_research_data(
        max_train_samples=N_SAMPLES_DISCOVERY,
        max_test_samples=1000,
        max_eval_samples=500,
        use_samanantar=True,
    )
    train = data_split.train
    prompts = data_split.steering_prompts[:N_STEERING_EVAL]

    # Define target languages:
    #   - Indic Indo-Aryan: HI, BN, UR
    #   - Indic Dravidian: TA, TE, KN, ML (complete coverage)
    #   - Non-Indic controls: DE, AR
    targets = ["hi", "bn", "ta", "te", "kn", "ml", "ur", "de", "ar"]
    methods = ["dense", "activation_diff", "monolinguality", "random"]

    # Use centralized LANG_TO_SCRIPT mapping from config for consistency
    lang_to_script = LANG_TO_SCRIPT

    # Load model (2B base with residual SAEs)
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in str(getattr(model, "model_id", "")).lower() else ""

    # ------------------------------------------------------------------
    # Semantic reference mode
    # ------------------------------------------------------------------
    # By default we measure prompt-faithfulness: LaBSE(prompt, output).
    # Set SEMANTIC_REFERENCE=baseline to instead measure preservation
    # relative to unsteered baseline outputs: LaBSE(baseline, steered).
    semantic_reference_mode = os.environ.get("SEMANTIC_REFERENCE", "prompt").strip().lower()
    baseline_refs: Dict[str, str] = {}
    if semantic_reference_mode in ("baseline", "base"):
        print("[exp9] SEMANTIC_REFERENCE=baseline: computing baseline references once.")
        for p in tqdm(prompts, desc="Baseline refs", leave=False):
            baseline_refs[p] = model.generate(p, max_new_tokens=64)
        semantic_reference_mode = "baseline"
    else:
        semantic_reference_mode = "prompt"

    all_results: Dict[str, Dict] = {}
    # For calibrated judge summaries, we keep a per-language pool of
    # SteeringEvalResult objects that actually have judge outputs.
    judge_pools: Dict[str, List] = {t: [] for t in targets}

    for target in targets:
        if target not in train or "en" not in train:
            print(f"Skipping {target}: missing train data.")
            continue

        print(f"\n=== Target language: {target} ===")
        lang_results = {"layers": {}}
        # Collect all pairwise tests (all layers/strengths) for a single FDR pass per language
        all_pairwise_tests = []

        for layer in TARGET_LAYERS:
            print(f"\n  Layer {layer}...")
            layer_results = {}
            success_by_strength: Dict[float, Dict[str, List[int]]] = {}

            for method in methods:
                print(f"    Method: {method}")
                try:
                    vec = build_steering_vector(
                        method=method,
                        model=model,
                        train_data=train,
                        layer=layer,
                        target_lang=target,
                        source_lang="en",
                    )
                except Exception as e:
                    print(f"      Skipping method {method} at layer {layer}: {e}")
                    continue

                # Evaluate steering across strengths
                method_results = {}
                for strength in STEERING_STRENGTHS:
                    outputs = []
                    for idx, p in enumerate(
                        tqdm(
                            prompts,
                            desc=f"{target}-L{layer}-{method}@{strength}",
                            leave=False,
                        )
                    ):
                        gen = model.generate_with_steering(
                            p,
                            layer=layer,
                            steering_vector=vec,
                            strength=strength,
                            max_new_tokens=64,
                        )
                        use_judge_here = bool(judge_stride) and (idx % judge_stride == 0)
                        ref = baseline_refs.get(p) if baseline_refs else None
                        res = evaluate_steering_output(
                                p,
                                gen,
                                method=method,
                                strength=strength,
                                layer=layer,
                                semantic_reference=ref,
                                use_llm_judge=use_judge_here,
                                compute_semantics=True,
                                target_script=lang_to_script.get(target, "devanagari"),
                                judge_lang=target,
                            )
                        outputs.append(res)
                        # Store for calibrated judge summary later.
                        if res.llm_judge_raw is not None:
                            judge_pools.setdefault(target, []).append(res)

                    agg = aggregate_results(
                        outputs,
                        target_script=lang_to_script.get(target, "devanagari"),
                    )
                    # Per-prompt overall success for paired statistical tests.
                    per_prompt_success = [1 if r.overall_success else 0 for r in outputs]
                    success_by_strength.setdefault(strength, {})[method] = per_prompt_success
                    method_results[str(strength)] = {
                        "n_samples": agg.n_samples,
                        "success_rate_script": agg.success_rate,
                        "success_rate_script_semantic": agg.semantic_success_rate,
                        "avg_target_script_ratio": agg.avg_target_script_ratio,
                        "avg_semantic_similarity": agg.avg_semantic_similarity,
                        "degradation_rate": agg.degradation_rate,
                        "sensitivity": agg.sensitivity,
                        # Reviewer-proofing: bootstrap CIs and separate metrics
                        # are computed in the evaluator; persist them so the
                        # paper can cite exact intervals without re-running.
                        "separate_metrics": agg.separate_metrics.to_dict()
                        if agg.separate_metrics
                        else None,
                    }

                layer_results[method] = method_results

            # Pairwise method comparisons per strength with Holm-Bonferroni correction.
            layer_stats: Dict[str, Dict] = {}
            for strength, by_method in success_by_strength.items():
                if len(by_method) < 2:
                    continue
                try:
                    comps = compute_all_pairwise_comparisons(
                        by_method,
                        test_type="wilcoxon",
                        correction="holm",
                    )
                    serialized = []
                    for comp in comps.get("comparisons", []):
                        r = comp["result"]
                        entry = {
                            "method1": comp["method1"],
                            "method2": comp["method2"],
                            "p_value": r.p_value,
                            "adjusted_p": comp["adjusted_p"],
                            "significant_uncorrected": r.significant_at_05,
                            "significant_corrected": comp["significant_corrected"],
                            "effect_size": r.effect_size,
                            "effect_size_name": r.effect_size_name,
                        }
                        serialized.append(entry)
                        all_pairwise_tests.append(entry)
                    layer_stats[str(strength)] = {
                        "comparisons": serialized,
                        "correction_method": comps.get("correction_method", "holm"),
                        "summary": comps.get("summary", {}),
                    }
                except Exception as e:
                    layer_stats[str(strength)] = {"error": str(e)}

            if layer_stats:
                layer_results["stats"] = layer_stats

            lang_results["layers"][str(layer)] = layer_results

        all_results[target] = lang_results

        # Apply a single Benjamini-Hochberg FDR correction across ALL pairwise
        # tests for this language (all layers + strengths) to keep the overall
        # false discovery rate controlled for “best layer/method” claims.
        if all_pairwise_tests:
            p_vals = [c["p_value"] for c in all_pairwise_tests if c.get("p_value") is not None]
            if p_vals:
                holm = holm_bonferroni_correction(p_vals)
                idx = 0
                n_sig = 0
                for comp in all_pairwise_tests:
                    if comp.get("p_value") is None:
                        continue
                    adj_p = holm["adjusted_p_values"][idx]
                    sig = holm["significant"][idx]
                    comp["p_value_holm_global"] = adj_p
                    comp["significant_holm_global"] = sig
                    n_sig += 1 if sig else 0
                    idx += 1
                lang_results["global_familywise_correction"] = {
                    "method": "holm_bonferroni",
                    "n_tests": len(p_vals),
                    "n_significant": n_sig,
                    "alpha": 0.05,
                }

    # Attach semantic truncation stats and reference mode per language so plots
    # still treat top-level keys as languages.
    sem_stats = get_semantic_truncation_stats()
    for lang in all_results:
        all_results[lang]["semantic_reference"] = semantic_reference_mode
        all_results[lang]["semantic_truncation_stats"] = sem_stats

    # Save full sweep results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp9_layer_sweep_steering{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Calibrated judge summary per language (reusing Exp11 statistics)
    # ------------------------------------------------------------------
    cal_table = load_judge_calibration_table()
    if not cal_table:
        print(
            "[exp9] Warning: no judge calibration statistics found. "
            "Structural metrics remain valid, but calibrated judge summaries "
            "will be omitted. Run Exp11 first to enable calibrated judge stats."
        )
    judge_summary: Dict[str, Dict] = {}
    for lang, pool in judge_pools.items():
        if lang not in cal_table:
            print(f"[exp9] Skipping judge calibration for {lang}: no per-language calibration stats.")
            continue
        cj = calibrated_judge_from_results(pool, lang=lang, calibration_table=cal_table)
        if cj is None:
            continue
        print(
            f"[exp9] Calibrated judge ({lang}): raw={cj.raw_accuracy:.3f}, "
            f"corrected={cj.corrected_accuracy:.3f}, "
            f"CI=({cj.confidence_interval[0]:.3f}, {cj.confidence_interval[1]:.3f}), "
            f"n_test={cj.n_test}, n_calib_0={cj.n_calib_0}, n_calib_1={cj.n_calib_1}"
        )
        judge_summary[lang] = {
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

    if judge_summary:
        js_path = out_dir / f"exp9_layer_sweep_judge_summary{suffix}.json"
        with open(js_path, "w") as f:
            json.dump({"languages": judge_summary}, f, indent=2)
        print(f"✓ Calibrated judge summary saved to {js_path}")


if __name__ == "__main__":
    run_layer_sweep()
