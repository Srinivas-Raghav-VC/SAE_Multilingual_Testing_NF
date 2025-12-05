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

from tqdm import tqdm
import torch

from config import (
    TARGET_LAYERS,
    STEERING_STRENGTHS,
    NUM_FEATURES,
    N_SAMPLES_DISCOVERY,
)
from data import load_research_data
from model import GemmaWithSAE
from experiments.exp2_steering import (
    get_activation_diff_features,
    get_monolinguality_features,
    get_random_features,
    construct_sae_steering_vector,
    construct_dense_steering_vector,
)
from evaluation_comprehensive import evaluate_steering_output, aggregate_results


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

    # Load research data (Samanantar + FLORES + QA + prompts)
    data_split = load_research_data(
        max_train_samples=N_SAMPLES_DISCOVERY,
        max_test_samples=1000,
        max_eval_samples=500,
        use_samanantar=True,
    )
    train = data_split.train
    prompts = data_split.steering_prompts

    # Define target languages:
    #   - Indic: HI, BN, TA, TE, UR
    #   - Non-Indic controls: DE, AR
    targets = ["hi", "bn", "ta", "te", "ur", "de", "ar"]
    methods = ["dense", "activation_diff", "monolinguality", "random"]

    # Load model (2B base with residual SAEs)
    model = GemmaWithSAE()
    model.load_model()

    all_results: Dict[str, Dict] = {}

    for target in targets:
        if target not in train or "en" not in train:
            print(f"Skipping {target}: missing train data.")
            continue

        print(f"\n=== Target language: {target} ===")
        lang_results = {"layers": {}}

        for layer in TARGET_LAYERS:
            print(f"\n  Layer {layer}...")
            layer_results = {}

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
                    for p in tqdm(
                        prompts[:50],
                        desc=f"{target}-L{layer}-{method}@{strength}",
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
                                method=method,
                                strength=strength,
                                layer=layer,
                                # Use LLM judge if available; semantic model optional
                                use_llm_judge=True,
                                compute_semantics=True,
                            )
                        )

                    agg = aggregate_results(outputs)
                    method_results[str(strength)] = {
                        "n_samples": agg.n_samples,
                        "success_rate_script": agg.success_rate,
                        "success_rate_script_semantic": agg.semantic_success_rate,
                        "avg_target_script_ratio": agg.avg_target_script_ratio,
                        "avg_semantic_similarity": agg.avg_semantic_similarity,
                        "degradation_rate": agg.degradation_rate,
                    }

                layer_results[method] = method_results

            lang_results["layers"][str(layer)] = layer_results

        all_results[target] = lang_results

    # Save full sweep results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp9_layer_sweep_steering.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    run_layer_sweep()

