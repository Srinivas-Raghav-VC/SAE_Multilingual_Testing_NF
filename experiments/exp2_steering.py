"""Experiment 2: Steering Method Comparison (H2)

H2: Attribution-selected features outperform activation-selected by ≥5%

NOTE: True "attribution" requires paired completions from the same prompt,
which is impractical. We use "activation difference on parallel texts" as 
a practical approximation:
- Compute mean activation per feature for Hindi texts
- Compute mean activation per feature for English texts
- Use difference to select features

Methods compared:
1. Activation-diff: Features with largest EN→HI activation difference
2. Monolinguality: Features with highest monolinguality score
3. Random: Random feature selection (baseline)
4. Dense: Mean activation difference in hidden space (no SAE)
"""

# Path fix for running from experiments/ directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from tqdm import tqdm
from pathlib import Path

from config import (
    TARGET_LAYERS,
    STEERING_STRENGTHS,
    NUM_FEATURES_OPTIONS,
    N_SAMPLES_DISCOVERY,
    N_SAMPLES_EVAL,
    MIN_SAMPLES_PER_LANGUAGE,
    MIN_PROMPTS_STEERING,
    STATISTICAL_CONFIG,
)
from data import load_research_data
from model import GemmaWithSAE
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
)
from stats import (
    bootstrap_ci,
    paired_ttest,
    wilcoxon_test,
    test_superiority_hypothesis,
    compare_methods_paired,
    holm_bonferroni_correction,
)


def get_activation_diff_features(model, texts_en, texts_hi, layer, top_k):
    """Select features by activation difference between languages."""
    sae = model.load_sae(layer)
    
    # Mean activation for English
    en_acts = []
    for text in texts_en:
        acts = model.get_sae_activations(text, layer)
        en_acts.append(acts.mean(dim=0))
    en_mean = torch.stack(en_acts).mean(dim=0)
    
    # Mean activation for Hindi
    hi_acts = []
    for text in texts_hi:
        acts = model.get_sae_activations(text, layer)
        hi_acts.append(acts.mean(dim=0))
    hi_mean = torch.stack(hi_acts).mean(dim=0)
    
    # Difference: positive = more active for Hindi
    diff = hi_mean - en_mean
    _, top_ids = diff.topk(top_k)
    
    return top_ids.tolist()


def get_monolinguality_features(model, texts_by_lang, layer, target_lang, top_k):
    """Select features by monolinguality score."""
    from experiments.exp1_feature_discovery import compute_activation_rates, compute_monolinguality
    
    # Compute rates
    rates_by_lang = {}
    for lang, texts in texts_by_lang.items():
        rates_by_lang[lang] = compute_activation_rates(model, texts[:200], layer)
    
    # Compute monolinguality
    mono = compute_monolinguality(rates_by_lang, target_lang)
    _, top_ids = mono.topk(top_k)
    
    return top_ids.tolist()


def get_random_features(sae, top_k, seed=42):
    """Select random features."""
    torch.manual_seed(seed)
    n_features = sae.cfg.d_sae
    return torch.randperm(n_features)[:top_k].tolist()


def construct_sae_steering_vector(model, layer, feature_ids):
    """Build steering vector from SAE decoder columns."""
    sae = model.load_sae(layer)
    # W_dec: (n_features, d_model) - each row is a decoder direction
    directions = sae.W_dec[feature_ids, :]  # (k, d_model)
    vector = directions.mean(dim=0)  # Average direction
    
    # Normalize to reasonable magnitude
    vector = vector / vector.norm() * (sae.cfg.d_in ** 0.5)
    return vector


def construct_dense_steering_vector(model, texts_en, texts_hi, layer):
    """Build dense steering vector (mean difference in hidden space)."""
    # Mean hidden state for English
    en_hidden = []
    for text in texts_en[:100]:
        h = model.get_hidden_states(text, layer)
        en_hidden.append(h.mean(dim=0))
    en_mean = torch.stack(en_hidden).mean(dim=0)
    
    # Mean hidden state for Hindi
    hi_hidden = []
    for text in texts_hi[:100]:
        h = model.get_hidden_states(text, layer)
        hi_hidden.append(h.mean(dim=0))
    hi_mean = torch.stack(hi_hidden).mean(dim=0)
    
    vector = hi_mean - en_mean
    vector = vector / vector.norm() * (vector.shape[0] ** 0.5)
    return vector


def run_baseline_experiment(model, prompts, target_script="devanagari"):
    """Run baseline (no-steering) to establish natural Hindi production rate.

    This is critical for interpreting steering results - we need to know
    how much Hindi the model produces WITHOUT intervention.
    """
    print("Running no-steering baseline...")
    eval_results = []

    for prompt in tqdm(prompts, desc="Baseline", leave=False):
        gen = model.generate(prompt, max_new_tokens=64)
        result = evaluate_steering_output(
            prompt=prompt,
            output=gen,
            method="baseline",
            strength=0.0,
            layer=0,
            target_script=target_script,
            compute_semantics=True,
            use_llm_judge=False,  # Skip LLM judge for speed in baseline
        )
        eval_results.append(result)

    agg = aggregate_results(eval_results, target_script=target_script)

    # Extract per-prompt success for statistical testing
    per_prompt_success = [1 if r.is_target_script else 0 for r in eval_results]

    return {
        "n_samples": agg.n_samples,
        "success_rate": agg.success_rate,
        "avg_script_ratio": agg.avg_target_script_ratio,
        "degradation_rate": agg.degradation_rate,
        "per_prompt_success": per_prompt_success,
        "generations": [r.output for r in eval_results[:5]],
        "separate_metrics": agg.separate_metrics.to_dict() if agg.separate_metrics else None,
    }


def run_steering_experiment(
    model, layer, prompts, steering_vector, strengths,
    target_script="devanagari", method_name="unknown"
):
    """Run steering at multiple strengths with comprehensive evaluation.

    Returns per-prompt success indicators for downstream statistical testing.
    """
    results = {}

    for strength in strengths:
        eval_results = []

        for prompt in tqdm(prompts, desc=f"{method_name}@{strength}", leave=False):
            gen = model.generate_with_steering(
                prompt, layer, steering_vector, strength,
                max_new_tokens=64
            )
            result = evaluate_steering_output(
                prompt=prompt,
                output=gen,
                method=method_name,
                strength=strength,
                layer=layer,
                target_script=target_script,
                compute_semantics=True,
                use_llm_judge=False,  # Speed optimization; full judge in exp9/exp11
            )
            eval_results.append(result)

        agg = aggregate_results(eval_results, target_script=target_script)

        # Per-prompt success for statistical testing
        per_prompt_success = [1 if r.is_target_script else 0 for r in eval_results]

        results[strength] = {
            "success_rate": agg.success_rate,
            "avg_script_ratio": agg.avg_target_script_ratio,
            "degradation_rate": agg.degradation_rate,
            "semantic_success_rate": agg.semantic_success_rate,
            "per_prompt_success": per_prompt_success,
            "generations": [r.output for r in eval_results[:5]],
            "separate_metrics": agg.separate_metrics.to_dict() if agg.separate_metrics else None,
        }

    return results


def main():
    # Load model
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data via unified research loader so that:
    # - training texts (for steering vectors) come from the train split
    #   (Samanantar + FLORES, deduped against test),
    # - evaluation prompts come from held-out FLORES EN + EVAL_PROMPTS.
    print("Loading research data for Exp2 (steering sanity check)...")
    data_split = load_research_data()
    train_data = data_split.train
    texts_en = train_data.get("en", [])
    texts_hi = train_data.get("hi", [])
    prompts_all = data_split.steering_prompts
    
    if not texts_en or not texts_hi:
        print("ERROR: Need training data for en and hi in Exp2.")
        return
    
    # Research rigor: validate minimum sample sizes
    if len(texts_en) < MIN_SAMPLES_PER_LANGUAGE:
        print(
            f"[exp2] WARNING: Only {len(texts_en)} English samples "
            f"(recommend >= {MIN_SAMPLES_PER_LANGUAGE} for reliable steering vectors)"
        )
    if len(texts_hi) < MIN_SAMPLES_PER_LANGUAGE:
        print(
            f"[exp2] WARNING: Only {len(texts_hi)} Hindi samples "
            f"(recommend >= {MIN_SAMPLES_PER_LANGUAGE} for reliable steering vectors)"
        )
    
    # Evaluation prompts
    #
    # Earlier revisions hard‑coded 5–10 English prompts here, which made
    # success rates in this experiment extremely noisy (each example changed
    # the rate by 10–20%). To get statistically meaningful comparisons
    # between steering methods we now use a substantially larger prompt
    # set drawn from FLORES English sentences, capped by N_SAMPLES_EVAL.
    #
    # This keeps Exp2 as a lightweight but numerically sensible sanity
    # check; the more exhaustive method/layer analysis is performed in Exp9.
    num_prompts = min(N_SAMPLES_EVAL, len(prompts_all))
    if num_prompts < 20:
        print(
            f"[exp2] Warning: only {num_prompts} prompts available; "
            "consider increasing N_SAMPLES_EVAL for more stable estimates."
        )
    test_prompts = prompts_all[:num_prompts]
    
    # Evaluate a subset of layers for steering (early, mid, late)
    layers_to_test = [l for l in TARGET_LAYERS if l in {5, 13, 20, 24}]
    if not layers_to_test:
        layers_to_test = [13]
    
    num_features = 25
    all_results = {"num_features": num_features, "layers": {}}
    
    # Run baseline ONCE (no steering) to establish natural Hindi production rate
    print("\n" + "=" * 60)
    print("BASELINE: No-steering control")
    print("=" * 60)
    baseline_results = run_baseline_experiment(model, test_prompts)
    all_results["baseline"] = baseline_results
    print(f"Baseline Hindi success rate: {baseline_results['success_rate']:.1%}")

    for layer in layers_to_test:
        print(f"\n{'=' * 60}")
        print(f"Layer {layer} - Steering Experiments")
        print("=" * 60)
        layer_results = {"methods": {}}

        # Method 1: Activation difference
        print("\n1. Activation-diff features...")
        act_diff_feats = get_activation_diff_features(model, texts_en, texts_hi, layer, num_features)
        act_diff_vec = construct_sae_steering_vector(model, layer, act_diff_feats)
        layer_results["methods"]["activation_diff"] = run_steering_experiment(
            model, layer, test_prompts, act_diff_vec, STEERING_STRENGTHS,
            method_name="activation_diff"
        )

        # Method 2: Monolinguality
        print("\n2. Monolinguality features...")
        mono_feats = get_monolinguality_features(model, train_data, layer, "hi", num_features)
        mono_vec = construct_sae_steering_vector(model, layer, mono_feats)
        layer_results["methods"]["monolinguality"] = run_steering_experiment(
            model, layer, test_prompts, mono_vec, STEERING_STRENGTHS,
            method_name="monolinguality"
        )

        # Method 3: Random
        print("\n3. Random features...")
        sae = model.load_sae(layer)
        random_feats = get_random_features(sae, num_features)
        random_vec = construct_sae_steering_vector(model, layer, random_feats)
        layer_results["methods"]["random"] = run_steering_experiment(
            model, layer, test_prompts, random_vec, STEERING_STRENGTHS,
            method_name="random"
        )

        # Method 4: Dense (no SAE)
        print("\n4. Dense steering...")
        dense_vec = construct_dense_steering_vector(model, texts_en, texts_hi, layer)
        layer_results["methods"]["dense"] = run_steering_experiment(
            model, layer, test_prompts, dense_vec, STEERING_STRENGTHS,
            method_name="dense"
        )

        all_results["layers"][str(layer)] = layer_results
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "exp2_steering_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Print summary across layers with CIs
    print(f"\n{'Layer':<8} {'Method':<20} {'Best Rate':<15} {'95% CI':<20} {'vs Baseline':<15}")
    print("-" * 80)

    baseline_rate = baseline_results["success_rate"]
    baseline_per_prompt = baseline_results["per_prompt_success"]

    for layer_str, layer_data in all_results["layers"].items():
        for method, data in layer_data["methods"].items():
            # Find best strength
            best_strength = max(data.keys(), key=lambda s: data[s]["success_rate"])
            best_data = data[best_strength]
            best_rate = best_data["success_rate"]

            # Compute bootstrap CI
            per_prompt = best_data.get("per_prompt_success", [])
            if per_prompt and len(per_prompt) >= 5:
                ci = bootstrap_ci(per_prompt)
                ci_str = f"[{ci.ci_low:.1%}, {ci.ci_high:.1%}]"
            else:
                ci_str = "N/A"

            # Improvement over baseline
            improvement = best_rate - baseline_rate
            imp_str = f"{improvement:+.1%}"

            print(f"{layer_str:<8} {method:<20} {best_rate:<15.1%} {ci_str:<20} {imp_str:<15}")

    # =========================================================================
    # H2 HYPOTHESIS TEST (with statistical significance)
    # =========================================================================
    print("\n" + "=" * 60)
    print("H2 HYPOTHESIS TEST: Activation-diff vs Monolinguality")
    print("=" * 60)

    # H2 comparison at mid-layer (13) if available, else first tested layer
    compare_layer = "13" if "13" in all_results["layers"] else next(iter(all_results["layers"].keys()))
    layer_data = all_results["layers"][compare_layer]["methods"]

    # Get best strength for each method
    act_best_str = max(layer_data["activation_diff"].keys(),
                       key=lambda s: layer_data["activation_diff"][s]["success_rate"])
    mono_best_str = max(layer_data["monolinguality"].keys(),
                        key=lambda s: layer_data["monolinguality"][s]["success_rate"])

    act_data = layer_data["activation_diff"][act_best_str]
    mono_data = layer_data["monolinguality"][mono_best_str]

    act_best = act_data["success_rate"]
    mono_best = mono_data["success_rate"]

    diff = (act_best - mono_best) * 100
    print(f"\nLayer {compare_layer} Results:")
    print(f"  Activation-diff: {act_best:.1%} (strength={act_best_str})")
    print(f"  Monolinguality:  {mono_best:.1%} (strength={mono_best_str})")
    print(f"  Difference:      {diff:+.1f}%")

    # Statistical test if we have per-prompt data
    act_per_prompt = act_data.get("per_prompt_success", [])
    mono_per_prompt = mono_data.get("per_prompt_success", [])

    statistical_results = {}
    if act_per_prompt and mono_per_prompt and len(act_per_prompt) == len(mono_per_prompt):
        # Paired test (same prompts)
        h2_test = test_superiority_hypothesis(
            treatment=act_per_prompt,
            control=mono_per_prompt,
            hypothesis_id="H2",
            description="Activation-diff outperforms monolinguality by ≥5%",
            margin=0.05,  # 5% superiority margin
            paired=True,
        )

        print(f"\nStatistical Test (paired Wilcoxon):")
        print(f"  {h2_test.interpretation}")
        print(f"  H2 Status: {'PASS' if h2_test.passed else 'FAIL'}")

        statistical_results["h2_test"] = h2_test.to_dict()

        # Also test each method against random baseline
        random_best_str = max(layer_data["random"].keys(),
                              key=lambda s: layer_data["random"][s]["success_rate"])
        random_per_prompt = layer_data["random"][random_best_str].get("per_prompt_success", [])

        if random_per_prompt and len(random_per_prompt) == len(act_per_prompt):
            print("\nMethod comparisons vs Random baseline:")
            methods_to_test = {
                "activation_diff": act_per_prompt,
                "monolinguality": mono_per_prompt,
            }
            comparisons = compare_methods_paired(
                {**methods_to_test, "random": random_per_prompt},
                baseline_method="random",
                test_type="wilcoxon"
            )
            for method, result in comparisons.items():
                print(f"  {method} vs random: p={result.p_value:.4f}, effect={result.effect_size:.3f} ({result.interpretation})")

            # Multiple comparison correction
            p_values = [comparisons[m].p_value for m in comparisons]
            corrected = holm_bonferroni_correction(p_values)
            print(f"\nHolm-Bonferroni corrected significance:")
            for i, method in enumerate(comparisons.keys()):
                sig = "***" if corrected["significant"][i] else "n.s."
                print(f"  {method}: p_adj={corrected['adjusted_p_values'][i]:.4f} {sig}")

            statistical_results["method_comparisons"] = {
                m: comparisons[m].to_dict() for m in comparisons
            }
            statistical_results["correction"] = corrected

    else:
        print(f"\nH2 Status (point estimate): {'PASS' if abs(diff) >= 5 else 'FAIL (difference < 5%)'}")

    # Save statistical results
    all_results["statistical_analysis"] = statistical_results


if __name__ == "__main__":
    main()
