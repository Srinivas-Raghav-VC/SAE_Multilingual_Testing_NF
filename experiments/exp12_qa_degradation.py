"""Experiment 12: Cross-Task Degradation on QA Benchmarks

Goal:
    Quantify how EN→target language steering affects QA performance
    (MLQA + IndicQA), addressing the \"steering makes models worse\" concern.

Design:
    - Use the same Gemma+SAE setup as other experiments.
    - For each target language with QA data (HI/DE/AR from MLQA and
      HI/BN/TA/TE from IndicQA):
        1. Build an EN→target steering vector using the same machinery as
           Exp9 (activation_diff by default).
        2. For each QA example, generate:
             (a) Baseline answer (no steering)
             (b) Steered answer (with best steering config)
        3. Evaluate both with the comprehensive evaluation stack:
             - Script dominance (target script)
             - Baseline-preservation similarity: LaBSE(baseline, steered)
             - Degradation (repetition)
             - Optional Gemini judge (language/faithfulness/coherence)
        4. Aggregate and compare baseline vs steered.

This gives a direct measure of cross-task degradation instead of only
looking at steering prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from typing import Dict, List, Tuple

from tqdm import tqdm
from reproducibility import seed_everything
import numpy as np

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    N_SAMPLES_EVAL,
    LANG_TO_SCRIPT,
    SEMANTIC_SIM_THRESHOLD,
)
from data import load_research_data
from model import GemmaWithSAE
from experiments.exp9_layer_sweep_steering import build_steering_vector
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
    semantic_similarity,
    load_judge_calibration_table,
    calibrated_judge_from_results,
    get_semantic_truncation_stats,
    estimate_power_paired,
)
from stats import wilcoxon_test, holm_bonferroni_correction, bootstrap_ci


# Use centralized LANG_TO_SCRIPT from config.py for consistency
# (imported above)


def load_best_steering_from_exp9(
    results_path: Path,
    target_lang: str,
) -> Tuple[int, float, str]:
    """Load best (layer, strength, method) for a target language from Exp9.

    If no result exists, fall back to:
        - last TARGET_LAYERS layer
        - strength 2.0
        - method \"activation_diff\"
    """
    default_layer = TARGET_LAYERS[-1]
    default_strength = 2.0
    default_method = "activation_diff"

    if not results_path.exists():
        return default_layer, default_strength, default_method

    with open(results_path) as f:
        sweep = json.load(f)

    if target_lang not in sweep:
        return default_layer, default_strength, default_method

    lang_data = sweep[target_lang].get("layers", {})
    best = (default_layer, default_strength, default_method, 0.0)

    for layer_str, methods in lang_data.items():
        layer = int(layer_str)
        for method, per_strength in methods.items():
            for strength_str, metrics in per_strength.items():
                strength = float(strength_str)
                # Prefer script+semantic success if available, else script-only
                ss = metrics.get("success_rate_script_semantic", None)
                if ss is None:
                    ss = metrics.get("success_rate_script", 0.0)
                if ss is None:
                    ss = 0.0
                if ss > best[3]:
                    best = (layer, strength, method, ss)

    return best[0], best[1], best[2]


def run_qa_eval_for_lang(
    model: GemmaWithSAE,
    qa_examples: List[Dict],
    target_lang: str,
    steering_vector,
    steering_layer: int,
    steering_strength: float,
    use_llm_judge: bool = True,
) -> Dict[str, Dict]:
    """Run baseline vs steered QA evaluation for a single language."""
    results_baseline = []
    results_steered = []
    baseline_preservation_sims: List[float] = []

    # Allow a QA-specific sample-size override without changing global config.
    # This helps scaling/power checks when compute permits.
    try:
        n_qa_samples = int(os.environ.get("N_QA_EVAL_SAMPLES", str(N_SAMPLES_EVAL)))
    except ValueError:
        n_qa_samples = N_SAMPLES_EVAL
    if n_qa_samples <= 0:
        n_qa_samples = N_SAMPLES_EVAL
    if "N_QA_EVAL_SAMPLES" in os.environ:
        print(f"[exp12] Using {n_qa_samples} QA examples per language (env override).")

    script = LANG_TO_SCRIPT.get(target_lang)
    if script is None:
        print(f"[exp12] WARNING: No script mapping for '{target_lang}', defaulting to 'devanagari'")
        script = "devanagari"

    for ex in tqdm(
        qa_examples[:n_qa_samples],
        desc=f"QA {target_lang} (baseline/steered)",
        leave=False,
    ):
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        gold = ex.get("answer", "")

        if not ctx or not q or not gold:
            continue

        prompt = (
            f"Context:\n{ctx}\n\n"
            f"Question:\n{q}\n\n"
            f"Answer in {target_lang.upper()}:"
        )

        # Baseline
        base_out = model.generate(prompt, max_new_tokens=64)
        res_base = evaluate_steering_output(
                prompt,
                base_out,
                method="baseline",
                strength=0.0,
                layer=-1,
                qa_references=gold,
                target_script=script,
                use_llm_judge=use_llm_judge,
                judge_lang=target_lang,
                compute_semantics=False,
            )
        results_baseline.append(res_base)

        # Steered
        steered_out = model.generate_with_steering(
            prompt,
            layer=steering_layer,
            steering_vector=steering_vector,
            strength=steering_strength,
            max_new_tokens=64,
        )
        res_steer = evaluate_steering_output(
                prompt,
                steered_out,
                method="steered",
                strength=steering_strength,
                layer=steering_layer,
                qa_references=gold,
                target_script=script,
                use_llm_judge=use_llm_judge,
                judge_lang=target_lang,
                compute_semantics=False,
            )
        results_steered.append(res_steer)

        # Semantic preservation proxy for QA: compare baseline vs steered outputs.
        # (Prompt-vs-answer similarity is not meaningful for long contexts.)
        baseline_preservation_sims.append(semantic_similarity(base_out, steered_out))

    agg_base = aggregate_results(results_baseline, target_script=script)
    agg_steer = aggregate_results(results_steered, target_script=script)

    pres_vals = [v for v in baseline_preservation_sims if v is not None and v >= 0.0]
    pres_mean = float(sum(pres_vals) / len(pres_vals)) if pres_vals else None
    pres_rate = (
        float(sum(1 for v in pres_vals if v >= SEMANTIC_SIM_THRESHOLD) / len(pres_vals))
        if pres_vals
        else None
    )

    def _mean_optional(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    qa_em_base = _mean_optional([r.qa_exact_match for r in results_baseline])
    qa_f1_base = _mean_optional([r.qa_f1 for r in results_baseline])
    qa_em_steer = _mean_optional([r.qa_exact_match for r in results_steered])
    qa_f1_steer = _mean_optional([r.qa_f1 for r in results_steered])

    # Calibrated judge summaries (if calibration stats exist for this lang).
    cal_table = load_judge_calibration_table()
    if not cal_table:
        print(
            "[exp12] Warning: no judge calibration statistics found. "
            "QA results will report only structural metrics; calibrated "
            "judge accuracies will be unavailable. Run Exp11 first to "
            "enable calibrated judge summaries."
        )
    judge_base = calibrated_judge_from_results(
        results_baseline, lang=target_lang, calibration_table=cal_table
    )
    judge_steer = calibrated_judge_from_results(
        results_steered, lang=target_lang, calibration_table=cal_table
    )

    def _judge_dict(cj):
        if cj is None:
            return None
        return {
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

    # ------------------------------------------------------------------
    # Paired statistical tests: baseline vs steered
    # ------------------------------------------------------------------
    paired_tests: Dict[str, Dict] = {}
    p_vals: List[float] = []
    names: List[str] = []

    # Overall success (binary)
    overall_base = [1 if r.overall_success else 0 for r in results_baseline]
    overall_steer = [1 if r.overall_success else 0 for r in results_steered]
    res_overall = wilcoxon_test(overall_steer, overall_base)
    paired_tests["overall_success"] = res_overall.to_dict()
    p_vals.append(res_overall.p_value)
    names.append("overall_success")

    # QA F1 (only if available for enough pairs)
    f1_pairs = [
        (b.qa_f1, s.qa_f1)
        for b, s in zip(results_baseline, results_steered)
        if b.qa_f1 is not None and s.qa_f1 is not None
    ]
    if len(f1_pairs) >= 5:
        f1_base = [b for b, _ in f1_pairs]
        f1_steer = [s for _, s in f1_pairs]
        res_f1 = wilcoxon_test(f1_steer, f1_base)
        paired_tests["qa_f1"] = res_f1.to_dict()
        p_vals.append(res_f1.p_value)
        names.append("qa_f1")

    # QA EM (binary)
    em_pairs = [
        (b.qa_exact_match, s.qa_exact_match)
        for b, s in zip(results_baseline, results_steered)
        if b.qa_exact_match is not None and s.qa_exact_match is not None
    ]
    if len(em_pairs) >= 5:
        em_base = [b for b, _ in em_pairs]
        em_steer = [s for _, s in em_pairs]
        res_em = wilcoxon_test(em_steer, em_base)
        paired_tests["qa_exact_match"] = res_em.to_dict()
        p_vals.append(res_em.p_value)
        names.append("qa_exact_match")

    # Holm-Bonferroni correction across the tested metrics.
    if p_vals:
        corrected = holm_bonferroni_correction(p_vals)
        for i, name in enumerate(names):
            paired_tests[name]["adjusted_p"] = corrected["adjusted_p_values"][i]
            paired_tests[name]["significant_corrected"] = corrected["significant"][i]
            # Add simple power estimate for QA F1 if available
            if name == "qa_f1":
                try:
                    paired_tests[name]["power"] = estimate_power_paired(f1_base, f1_steer)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Publication decision criteria (preregistered-style)
    # ------------------------------------------------------------------
    decision: Dict[str, object] = {}
    threshold = 0.02
    if len(f1_pairs) >= 5:
        diffs = [float(s - b) for b, s in f1_pairs]  # steered - baseline
        ci = bootstrap_ci(diffs, statistic=np.mean)
        mean_delta = float(np.mean(diffs))
        # "Harmful" only if we have evidence of a negative shift (CI excludes 0)
        harmful = (mean_delta < -threshold) and (ci.ci_high < 0.0)
        beneficial = (mean_delta > threshold) and (ci.ci_low > 0.0)
        decision["qa_f1"] = {
            "mean_delta": mean_delta,
            "ci_95_mean_delta": [ci.ci_low, ci.ci_high],
            "threshold_abs": threshold,
            "harmful": harmful,
            "beneficial": beneficial,
            "interpretation": (
                "harmful" if harmful else "beneficial" if beneficial else "inconclusive"
            ),
        }
    else:
        decision["qa_f1"] = {
            "status": "insufficient_data",
            "n_pairs": len(f1_pairs),
            "threshold_abs": threshold,
        }

    return {
        "baseline": {
            "n_samples": agg_base.n_samples,
            "success_rate_script": agg_base.success_rate,
            "success_rate_script_semantic": agg_base.semantic_success_rate,
            "avg_target_script_ratio": agg_base.avg_target_script_ratio,
            "avg_semantic_similarity": agg_base.avg_semantic_similarity,
            "degradation_rate": agg_base.degradation_rate,
            "qa_exact_match": qa_em_base,
            "qa_f1": qa_f1_base,
            "judge": _judge_dict(judge_base),
            "sensitivity": agg_base.sensitivity,
            "separate_metrics": agg_base.separate_metrics.to_dict()
            if agg_base.separate_metrics
            else None,
        },
        "steered": {
            "n_samples": agg_steer.n_samples,
            "success_rate_script": agg_steer.success_rate,
            "success_rate_script_semantic": agg_steer.semantic_success_rate,
            "avg_target_script_ratio": agg_steer.avg_target_script_ratio,
            "avg_semantic_similarity": agg_steer.avg_semantic_similarity,
            "degradation_rate": agg_steer.degradation_rate,
            "qa_exact_match": qa_em_steer,
            "qa_f1": qa_f1_steer,
            "judge": _judge_dict(judge_steer),
            "baseline_preservation_semantic_mean": pres_mean,
            "baseline_preservation_semantic_rate": pres_rate,
            "sensitivity": agg_steer.sensitivity,
            "separate_metrics": agg_steer.separate_metrics.to_dict()
            if agg_steer.separate_metrics
            else None,
        },
        "stats": paired_tests,
        "decision": decision,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 12: QA Degradation Under Steering")
    print("=" * 60)

    from config import SEED
    seed_everything(SEED)

    # Load research data (includes QA + steering prompts)
    data_split = load_research_data(
        max_train_samples=N_SAMPLES_DISCOVERY,
        max_test_samples=1000,
        max_eval_samples=N_SAMPLES_EVAL,
        use_samanantar=True,
    )

    train = data_split.train
    qa_eval = data_split.qa_eval or {}

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in str(getattr(model, "model_id", "")).lower() else ""

    # Load best steering configs from Exp9 if available
    exp9_path = Path("results") / f"exp9_layer_sweep_steering{suffix}.json"
    if not exp9_path.exists():
        exp9_path = Path("results") / "exp9_layer_sweep_steering.json"

    results: Dict[str, Dict] = {}

    # MLQA languages: EN/HI/DE/AR (but we only steer EN→target)
    mlqa = qa_eval.get("mlqa", {})
    for lang in ["hi", "de", "ar"]:
        if lang not in mlqa or lang not in train or "en" not in train:
            continue

        print(f"\n=== MLQA language: {lang} ===")
        layer, strength, method = load_best_steering_from_exp9(exp9_path, lang)
        print(f"Using steering config from Exp9 (or default): layer={layer}, strength={strength}, method={method}")

        vec = build_steering_vector(
            method=method,
            model=model,
            train_data=train,
            layer=layer,
            target_lang=lang,
            source_lang="en",
        )

        lang_results = run_qa_eval_for_lang(
            model,
            mlqa[lang],
            target_lang=lang,
            steering_vector=vec,
            steering_layer=layer,
            steering_strength=strength,
            use_llm_judge=True,
        )
        lang_results["steering_config"] = {
            "layer": layer,
            "strength": strength,
            "method": method,
        }
        # Publication-grade decision rule: classify QA impact
        qa_base = lang_results["baseline"].get("qa_f1")
        qa_steer = lang_results["steered"].get("qa_f1")
        if qa_base is not None and qa_steer is not None:
            delta = qa_steer - qa_base
            verdict = "no_effect"
            if delta < -0.02:
                verdict = "harmful"
            elif delta > 0.02:
                verdict = "improved"
            lang_results["qa_impact"] = {"delta_f1": delta, "verdict": verdict}
        results[f"mlqa_{lang}"] = lang_results

    # IndicQA languages: HI/BN/TA/TE
    indicqa = qa_eval.get("indicqa", {})
    for lang in ["hi", "bn", "ta", "te"]:
        if lang not in indicqa or lang not in train or "en" not in train:
            continue

        print(f"\n=== IndicQA language: {lang} ===")
        layer, strength, method = load_best_steering_from_exp9(exp9_path, lang)
        print(f"Using steering config from Exp9 (or default): layer={layer}, strength={strength}, method={method}")

        vec = build_steering_vector(
            method=method,
            model=model,
            train_data=train,
            layer=layer,
            target_lang=lang,
            source_lang="en",
        )

        lang_results = run_qa_eval_for_lang(
            model,
            indicqa[lang],
            target_lang=lang,
            steering_vector=vec,
            steering_layer=layer,
            steering_strength=strength,
            use_llm_judge=True,
        )
        lang_results["steering_config"] = {
            "layer": layer,
            "strength": strength,
            "method": method,
        }
        qa_base = lang_results["baseline"].get("qa_f1")
        qa_steer = lang_results["steered"].get("qa_f1")
        if qa_base is not None and qa_steer is not None:
            delta = qa_steer - qa_base
            verdict = "no_effect"
            if delta < -0.02:
                verdict = "harmful"
            elif delta > 0.02:
                verdict = "improved"
            lang_results["qa_impact"] = {"delta_f1": delta, "verdict": verdict}
        results[f"indicqa_{lang}"] = lang_results

    # Attach LaBSE truncation stats inside each task block to avoid breaking
    # downstream summaries that assume top-level keys are tasks.
    sem_stats = get_semantic_truncation_stats()
    for k in results:
        results[k]["semantic_truncation_stats"] = sem_stats

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp12_qa_degradation{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
