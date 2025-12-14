"""Experiment 11: Calibrated LLM-as-Judge Evaluation (Gemini)

Goal:
    Measure how biased Gemini is as a judge of Hindi (and optionally
    German) steering success, and compute bias-corrected estimates of
    success with confidence intervals.

Method (per language):
    1. Use Samanantar+FLORES via load_research_data(use_samanantar=True)
       to build an activation-diff steering vector EN→L at a fixed layer.
    2. Generate steered outputs for a set of prompts.
    3. Use our script+semantic evaluator (no LLM judge) to assign
       ground_truth_is_correct labels (proxy ground truth).
    4. Split into calibration set and test set.
    5. Run evaluate_with_calibrated_judge with Gemini as judge.

Notes:
    - This treats the script+semantic success metric as surrogate ground
      truth, following Lee et al. (we calibrate the judge relative to
      this metric, not human labels).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
import json
import os

from config import TARGET_LAYERS, N_SAMPLES_EVAL, STEERING_STRENGTHS, LANG_TO_SCRIPT, SEED
from data import load_research_data
from model import GemmaWithSAE
from steering_utils import (
    get_activation_diff_features,
    construct_sae_steering_vector,
)
from evaluation_comprehensive import (
    evaluate_steering_output,
    evaluate_with_calibrated_judge,
    llm_judge_gemini,
)
from reproducibility import seed_everything


def build_activation_diff_vector(model, train: Dict[str, List[str]], layer: int, target_lang: str, k: int = 25):
    """Build EN→target activation-diff steering vector at a given layer."""
    texts_en = train.get("en", [])
    texts_tgt = train.get(target_lang, [])
    if not texts_en or not texts_tgt:
        raise ValueError(f"Missing training data for en or {target_lang}")
    
    feats = get_activation_diff_features(model, texts_en, texts_tgt, layer, k)
    return construct_sae_steering_vector(model, layer, feats)


def run_calibration_for_language(lang: str, layer: int = 20, strength: float = 2.0):
    """Run calibration experiment for a single target language."""
    print(f"\n=== Judge calibration for language: {lang}, layer {layer}, strength {strength} ===")
    
    # Load research data (train + prompts)
    data_split = load_research_data(
        max_train_samples=5000,
        max_test_samples=1000,
        max_eval_samples=500,
        use_samanantar=True,
    )
    train = data_split.train
    prompts = data_split.steering_prompts[: max(N_SAMPLES_EVAL, 50)]
    
    # Load model
    model = GemmaWithSAE()
    model.load_model()
    
    # Build steering vector
    vec = build_activation_diff_vector(model, train, layer, lang, k=25)
    
    # Generate steered outputs + ground truth labels from script+semantics evaluator
    print("  Generating steered outputs for calibration/test...")
    outputs = []
    for p in prompts:
        gen = model.generate_with_steering(
            p,
            layer=layer,
            steering_vector=vec,
            strength=strength,
            max_new_tokens=64,
        )
        eval_res = evaluate_steering_output(
            prompt=p,
            output=gen,
            method="activation_diff",
            strength=strength,
            layer=layer,
            compute_semantics=True,
            use_llm_judge=False,
            target_script=LANG_TO_SCRIPT.get(lang, "devanagari"),
        )
        outputs.append(
            {
                "prompt": p,
                "output": gen,
                "ground_truth_is_correct": bool(eval_res.overall_success),
            }
        )
    
    if len(outputs) < 20:
        print("  Not enough outputs for calibration; skipping.")
        return None
    
    # Split into calibration and test sets. We aim for reasonably sized
    # sets (at least ~100 each) when data permits, to reduce variance in
    # the Lee et al. calibration estimates.
    split_idx = len(outputs) // 2
    calibration_set = outputs[:split_idx]
    test_outputs = outputs[split_idx:]
    
    print(f"  Calibration set size: {len(calibration_set)}")
    print(f"  Test set size: {len(test_outputs)}")
    
    # Run calibrated judge evaluation
    calib_result = evaluate_with_calibrated_judge(
        test_outputs=test_outputs,
        calibration_set=calibration_set,
        # Wrap the judge so it always uses the correct target language in
        # the prompt (important for multilingual calibration).
        judge_fn=lambda p, o: llm_judge_gemini(p, o, lang_code=lang),
        success_threshold=3,  # language score ≥3 counts as judge "correct"
    )
    
    return calib_result


def main():
    print("=" * 60)
    print("EXPERIMENT 11: Calibrated LLM-as-Judge Evaluation (Gemini)")
    print("=" * 60)

    seed_everything(SEED)
    
    # Choose a late-ish layer; 20 is typical steering layer
    # (this is not a hyperparam search, just a fixed probe)
    layer = 20 if 20 in TARGET_LAYERS else TARGET_LAYERS[-1]
    strength = 2.0
    
    # Languages for which we want calibrated judge stats. These cover the
    # languages where we currently use Gemini in other experiments.
    target_langs = ["hi", "ur", "bn", "ta", "te", "de", "ar"]
    results = {}
    
    for lang in target_langs:
        try:
            calib = run_calibration_for_language(lang, layer=layer, strength=strength)
            if calib is not None:
                print(
                    f"  [{lang}] raw={calib.raw_accuracy:.3f}, "
                    f"corrected={calib.corrected_accuracy:.3f}, "
                    f"CI=({calib.confidence_interval[0]:.3f}, {calib.confidence_interval[1]:.3f}), "
                    f"n_test={calib.n_test}, n_calib_0={calib.n_calib_0}, n_calib_1={calib.n_calib_1}"
                )
                results[lang] = {
                    "raw_accuracy": calib.raw_accuracy,
                    "corrected_accuracy": calib.corrected_accuracy,
                    "confidence_interval": calib.confidence_interval,
                    "q0": calib.q0,
                    "q1": calib.q1,
                    "n_test": calib.n_test,
                    "n_calib_0": calib.n_calib_0,
                    "n_calib_1": calib.n_calib_1,
                }
        except Exception as e:
            print(f"  Calibration for {lang} failed: {e}")
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    suffix = "_9b" if str(os.environ.get("USE_9B", "0")).lower() in ("1", "true", "yes") else ""
    out_path = out_dir / f"exp11_judge_calibration{suffix}.json"
    
    with open(out_path, "w") as f:
        json.dump(
            {
                "layer": layer,
                "strength": strength,
                "languages": results,
            },
            f,
            indent=2,
        )
    
    print(f"\n✓ Judge calibration results saved to {out_path}")


if __name__ == "__main__":
    main()
