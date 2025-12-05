"""Experiment 16: Code-Mixed and Noisy Prompt Robustness

Goal
----
Test how robust EN→HI steering is to code-mixed and noisy inputs:
  - Clean English prompts (baseline condition).
  - Code-mixed prompts that mix English + Devanagari + Hinglish tokens.

Design
------
1. Build an EN→HI activation-diff steering vector at a late layer.
2. Construct three prompt sets:
     a) clean_en: EVAL_PROMPTS (English).
     b) en_plus_deva: EVAL_PROMPTS with appended Devanagari tokens.
     c) hinglish_mix: EVAL_PROMPTS with appended Latin Hinglish tokens
        plus some Devanagari words.
3. For each prompt set:
     - Evaluate baseline vs steered outputs:
         * target script: Devanagari
         * metrics: script success, degradation.
4. Compare whether steering effectiveness and degradation are similar
   across clean and code-mixed/noisy prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
import json
import random

from tqdm import tqdm

from config import TARGET_LAYERS, N_SAMPLES_EVAL, EVAL_PROMPTS
from data import load_research_data
from model import GemmaWithSAE
from experiments.exp9_layer_sweep_steering import build_steering_vector
from evaluation_comprehensive import evaluate_steering_output, aggregate_results


DEVANAGARI_WORDS = [" दिल्ली", " भारत", " नमस्ते", " दोस्त", " परिवार"]
HINGLISH_TOKENS = ["yaar", "bhai", "mast", "dilli", "desi", "chai"]


def make_en_plus_deva(prompts: List[str]) -> List[str]:
    """Append random Devanagari words to English prompts."""
    out = []
    for p in prompts:
        w = random.choice(DEVANAGARI_WORDS)
        out.append(p + w)
    return out


def make_hinglish_mix(prompts: List[str]) -> List[str]:
    """Add Hinglish + Devanagari noise to English prompts."""
    out = []
    for p in prompts:
        latin = random.choice(HINGLISH_TOKENS)
        deva = random.choice(DEVANAGARI_WORDS)
        out.append(f"{p} {latin}{deva}")
    return out


def eval_prompt_set(
    model: GemmaWithSAE,
    prompts: List[str],
    layer: int,
    steering_vector,
    name: str,
) -> Dict[str, float]:
    """Evaluate baseline vs steered on a given prompt set."""
    base_results = []
    steer_results = []

    # Baseline
    for p in prompts:
        out = model.generate(p, max_new_tokens=64)
        base_results.append(
            evaluate_steering_output(
                p,
                out,
                method=f"{name}_baseline",
                strength=0.0,
                layer=layer,
                target_script="devanagari",
                compute_semantics=False,
                use_llm_judge=False,
            )
        )
    base_agg = aggregate_results(base_results, target_script="devanagari")

    # Steered
    strength = 2.0
    for p in tqdm(prompts, desc=f"{name}_steered@{strength}", leave=False):
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
                method=f"{name}_steered",
                strength=strength,
                layer=layer,
                target_script="devanagari",
                compute_semantics=False,
                use_llm_judge=False,
            )
        )
    steer_agg = aggregate_results(steer_results, target_script="devanagari")

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
    print("EXPERIMENT 16: Code-mix and Noise Robustness (EN→HI steering)")
    print("=" * 60)

    # Use a late-ish layer for steering
    layer_candidates = [l for l in TARGET_LAYERS if l >= 16]
    layer = layer_candidates[-1] if layer_candidates else TARGET_LAYERS[-1]
    print(f"Using layer {layer} for EN→HI steering.")

    # Load research data (for EN/HI train)
    data_split = load_research_data(
        max_train_samples=5000,
        max_test_samples=1000,
        max_eval_samples=N_SAMPLES_EVAL,
        use_samanantar=True,
    )
    train = data_split.train

    if "en" not in train or "hi" not in train:
        print("ERROR: Need en and hi training data.")
        return

    # Load model
    model = GemmaWithSAE()
    model.load_model()

    # Build EN→HI steering vector
    print("Building EN→HI steering vector...")
    vec_en_to_hi = build_steering_vector(
        method="activation_diff",
        model=model,
        train_data=train,
        layer=layer,
        target_lang="hi",
        source_lang="en",
    )

    # Build prompt sets
    base_prompts = EVAL_PROMPTS[: max(N_SAMPLES_EVAL, 50)]
    en_plus_deva = make_en_plus_deva(base_prompts)
    hinglish_mix = make_hinglish_mix(base_prompts)

    print(f"Using {len(base_prompts)} prompts per condition.")

    results = {}
    results["clean_en"] = eval_prompt_set(
        model, base_prompts, layer=layer, steering_vector=vec_en_to_hi, name="clean_en"
    )
    results["en_plus_deva"] = eval_prompt_set(
        model, en_plus_deva, layer=layer, steering_vector=vec_en_to_hi, name="en_plus_deva"
    )
    results["hinglish_mix"] = eval_prompt_set(
        model, hinglish_mix, layer=layer, steering_vector=vec_en_to_hi, name="hinglish_mix"
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp16_code_mix_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()

