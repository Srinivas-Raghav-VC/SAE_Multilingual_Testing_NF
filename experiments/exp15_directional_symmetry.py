"""Experiment 15: Directional Symmetry of Language Steering (EN↔Indic)

Goal
----
Test whether steering is symmetric in "difficulty" between:
  - EN→Indic (English prompts, steer into Indic script),
  - Indic→EN (Indic prompts, steer into English/Latin script),
for multiple Indic languages and a fixed late layer.

Design
------
For each Indic language L in {hi, bn, ta, te, ur}:
  1. Build EN→L activation-diff steering vector at layer ℓ.
  2. Build L→EN activation-diff steering vector at the same layer ℓ.
  3. EN→L:
       - Prompts: steering_prompts (English).
       - Evaluate baseline vs steered with target_script = script(L).
  4. L→EN:
       - Prompts: sample sentences in L from train data.
       - Evaluate baseline vs steered with target_script = "latin".

We compare success_rate_script (and degradation) for EN→L vs L→EN.
If the "language-agnostic" space is symmetric, we expect similar
steering effectiveness depths; if it's English-anchored, EN→L may be
easier than L→EN or vice versa.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List
import json

from tqdm import tqdm

from config import TARGET_LAYERS, N_SAMPLES_EVAL
from data import load_research_data
from model import GemmaWithSAE
from experiments.exp9_layer_sweep_steering import build_steering_vector
from evaluation_comprehensive import evaluate_steering_output, aggregate_results


INDIC_LANGS = ["hi", "bn", "ta", "te", "ur"]

LANG_TO_SCRIPT = {
    "hi": "devanagari",
    "bn": "bengali",
    "ta": "tamil",
    "te": "telugu",
    "ur": "arabic",
}


def eval_direction(
    model: GemmaWithSAE,
    prompts: List[str],
    layer: int,
    steering_vector,
    target_script: str,
) -> Dict[str, float]:
    """Evaluate baseline vs steered on a set of prompts."""
    base_results = []
    steer_results = []

    # Baseline
    for p in prompts:
        out = model.generate(p, max_new_tokens=64)
        base_results.append(
            evaluate_steering_output(
                p,
                out,
                method="baseline",
                strength=0.0,
                layer=layer,
                target_script=target_script,
                compute_semantics=False,
                use_llm_judge=False,
            )
        )
    base_agg = aggregate_results(base_results, target_script=target_script)

    # Steered (use a single strength, 2.0, for simplicity)
    strength = 2.0
    for p in tqdm(prompts, desc=f"steered@{strength}", leave=False):
        out = model.generate_with_steering(
            p,
            layer=layer,
            steering_vector=steering_vector,
            strength=strength,
            max_new_tokens=64,
        )
        steer_results.append(
            evaluate_steering_output(
                p,
                out,
                method="activation_diff",
                strength=strength,
                layer=layer,
                target_script=target_script,
                compute_semantics=False,
                use_llm_judge=False,
            )
        )
    steer_agg = aggregate_results(steer_results, target_script=target_script)

    return {
        "baseline_success_script": base_agg.success_rate,
        "baseline_degradation_rate": base_agg.degradation_rate,
        "steered_success_script": steer_agg.success_rate,
        "steered_degradation_rate": steer_agg.degradation_rate,
        "delta_success_script": steer_agg.success_rate - base_agg.success_rate,
        "delta_degradation_rate": steer_agg.degradation_rate - base_agg.degradation_rate,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 15: Directional Symmetry of Steering (EN↔Indic)")
    print("=" * 60)

    # Use a late-ish layer where steering is usually effective
    layer_candidates = [l for l in TARGET_LAYERS if l >= 16]
    layer = layer_candidates[-1] if layer_candidates else TARGET_LAYERS[-1]
    print(f"Using layer {layer} for symmetry analysis.")

    # Load research data
    data_split = load_research_data(
        max_train_samples=5000,
        max_test_samples=1000,
        max_eval_samples=N_SAMPLES_EVAL,
        use_samanantar=True,
    )
    train = data_split.train
    steering_prompts = data_split.steering_prompts[: max(N_SAMPLES_EVAL, 50)]

    # Load model once
    model = GemmaWithSAE()
    model.load_model()

    results: Dict[str, Dict] = {}

    for lang in INDIC_LANGS:
        print(f"\n=== Language: {lang} ===")

        if "en" not in train or lang not in train:
            print(f"  Skipping {lang}: missing train data.")
            continue

        # EN -> L (English prompts, target script = script(L))
        print("  Building EN→L steering vector...")
        vec_en_to_l = build_steering_vector(
            method="activation_diff",
            model=model,
            train_data=train,
            layer=layer,
            target_lang=lang,
            source_lang="en",
        )

        prompts_en = steering_prompts
        script_l = LANG_TO_SCRIPT.get(lang, "devanagari")
        print("  Evaluating EN→L...")
        en_to_l_res = eval_direction(
            model,
            prompts_en,
            layer=layer,
            steering_vector=vec_en_to_l,
            target_script=script_l,
        )

        # L -> EN (Indic prompts, target script = latin)
        print("  Building L→EN steering vector...")
        vec_l_to_en = build_steering_vector(
            method="activation_diff",
            model=model,
            train_data=train,
            layer=layer,
            target_lang="en",
            source_lang=lang,
        )

        prompts_l = train[lang][: len(steering_prompts)]
        print("  Evaluating L→EN...")
        l_to_en_res = eval_direction(
            model,
            prompts_l,
            layer=layer,
            steering_vector=vec_l_to_en,
            target_script="latin",
        )

        results[lang] = {
            "layer": layer,
            "en_to_l": en_to_l_res,
            "l_to_en": l_to_en_res,
        }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp15_directional_symmetry.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()

