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
             - Semantic similarity to gold answer (LaBSE)
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
from typing import Dict, List, Tuple

from tqdm import tqdm

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    N_SAMPLES_EVAL,
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
)


LANG_TO_SCRIPT = {
    "hi": "devanagari",
    "bn": "bengali",
    "ta": "tamil",
    "te": "telugu",
    "de": "latin",
    "ar": "arabic",
}


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

    script = LANG_TO_SCRIPT.get(target_lang, "devanagari")

    for ex in tqdm(
        qa_examples[:N_SAMPLES_EVAL],
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
                reference=gold,
                target_script=script,
                use_llm_judge=use_llm_judge,
                compute_semantics=True,
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
                reference=gold,
                target_script=script,
                use_llm_judge=use_llm_judge,
                compute_semantics=True,
            )
        results_steered.append(res_steer)

    agg_base = aggregate_results(results_baseline, target_script=script)
    agg_steer = aggregate_results(results_steered, target_script=script)

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

    return {
        "baseline": {
            "n_samples": agg_base.n_samples,
            "success_rate_script": agg_base.success_rate,
            "success_rate_script_semantic": agg_base.semantic_success_rate,
            "avg_target_script_ratio": agg_base.avg_target_script_ratio,
            "avg_semantic_similarity": agg_base.avg_semantic_similarity,
            "degradation_rate": agg_base.degradation_rate,
            "judge": _judge_dict(judge_base),
        },
        "steered": {
            "n_samples": agg_steer.n_samples,
            "success_rate_script": agg_steer.success_rate,
            "success_rate_script_semantic": agg_steer.semantic_success_rate,
            "avg_target_script_ratio": agg_steer.avg_target_script_ratio,
            "avg_semantic_similarity": agg_steer.avg_semantic_similarity,
            "degradation_rate": agg_steer.degradation_rate,
            "judge": _judge_dict(judge_steer),
        },
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 12: QA Degradation Under Steering")
    print("=" * 60)

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

    # Load best steering configs from Exp9 if available
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
        results[f"indicqa_{lang}"] = lang_results

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp12_qa_degradation.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
