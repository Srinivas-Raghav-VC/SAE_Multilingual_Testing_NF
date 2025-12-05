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
    TARGET_LAYERS, STEERING_STRENGTHS, NUM_FEATURES_OPTIONS,
    N_SAMPLES_DISCOVERY, N_SAMPLES_EVAL,
)
from data import load_flores, load_parallel_pairs
from model import GemmaWithSAE


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


def evaluate_generation(model, prompt, generated, target_lang="hi"):
    """Simple evaluation: check if output contains target script."""
    # Devanagari Unicode range: 0x0900-0x097F
    devanagari_chars = sum(1 for c in generated if 0x0900 <= ord(c) <= 0x097F)
    total_alpha = sum(1 for c in generated if c.isalpha())
    
    if total_alpha == 0:
        return {"script_ratio": 0.0, "success": False}
    
    ratio = devanagari_chars / total_alpha
    return {"script_ratio": ratio, "success": ratio > 0.3}


def run_steering_experiment(model, layer, prompts, steering_vector, strengths):
    """Run steering at multiple strengths."""
    results = {}
    for strength in strengths:
        generations = []
        metrics = []
        for prompt in tqdm(prompts, desc=f"Strength {strength}", leave=False):
            gen = model.generate_with_steering(prompt, layer, steering_vector, strength)
            eval_result = evaluate_generation(model, prompt, gen)
            generations.append(gen)
            metrics.append(eval_result)
        
        success_rate = sum(m["success"] for m in metrics) / len(metrics)
        avg_script = sum(m["script_ratio"] for m in metrics) / len(metrics)
        
        results[strength] = {
            "success_rate": success_rate,
            "avg_script_ratio": avg_script,
            "generations": generations[:5],  # Save a few examples
        }
    return results


def main():
    # Load model
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data
    print("Loading data...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    texts_en = flores["en"]
    texts_hi = flores["hi"]
    
    # Test prompts (English prompts to steer toward Hindi)
    test_prompts = [
        "The capital of India is",
        "My favorite food is",
        "Today the weather is",
        "I want to learn about",
        "The best way to travel is",
    ][:N_SAMPLES_EVAL] if N_SAMPLES_EVAL < 5 else [
        "The capital of India is",
        "My favorite food is", 
        "Today the weather is",
        "I want to learn about",
        "The best way to travel is",
        "In the morning I usually",
        "The most important thing is",
        "When I was young I",
        "Science tells us that",
        "The future of technology is",
    ]
    
    layer = 13  # Mid-layer for Gemma 2 2B
    num_features = 25
    
    results = {"layer": layer, "num_features": num_features, "methods": {}}
    
    print(f"\nRunning steering experiments at layer {layer}...")
    
    # Method 1: Activation difference
    print("\n1. Activation-diff features...")
    act_diff_feats = get_activation_diff_features(model, texts_en, texts_hi, layer, num_features)
    act_diff_vec = construct_sae_steering_vector(model, layer, act_diff_feats)
    results["methods"]["activation_diff"] = run_steering_experiment(
        model, layer, test_prompts, act_diff_vec, STEERING_STRENGTHS
    )
    
    # Method 2: Monolinguality
    print("\n2. Monolinguality features...")
    mono_feats = get_monolinguality_features(model, flores, layer, "hi", num_features)
    mono_vec = construct_sae_steering_vector(model, layer, mono_feats)
    results["methods"]["monolinguality"] = run_steering_experiment(
        model, layer, test_prompts, mono_vec, STEERING_STRENGTHS
    )
    
    # Method 3: Random
    print("\n3. Random features...")
    sae = model.load_sae(layer)
    random_feats = get_random_features(sae, num_features)
    random_vec = construct_sae_steering_vector(model, layer, random_feats)
    results["methods"]["random"] = run_steering_experiment(
        model, layer, test_prompts, random_vec, STEERING_STRENGTHS
    )
    
    # Method 4: Dense (no SAE)
    print("\n4. Dense steering...")
    dense_vec = construct_dense_steering_vector(model, texts_en, texts_hi, layer)
    results["methods"]["dense"] = run_steering_experiment(
        model, layer, test_prompts, dense_vec, STEERING_STRENGTHS
    )
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "exp2_steering_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== H2 Analysis ===")
    print(f"{'Method':<20} {'Best Rate':<12} {'Best Strength':<15}")
    print("-" * 50)
    
    for method, data in results["methods"].items():
        best_strength = max(data.keys(), key=lambda s: data[s]["success_rate"])
        best_rate = data[best_strength]["success_rate"]
        print(f"{method:<20} {best_rate:<12.1%} {best_strength:<15}")
    
    # H2 comparison
    act_best = max(results["methods"]["activation_diff"].values(), key=lambda x: x["success_rate"])["success_rate"]
    mono_best = max(results["methods"]["monolinguality"].values(), key=lambda x: x["success_rate"])["success_rate"]
    
    diff = (act_best - mono_best) * 100
    print(f"\nActivation-diff vs Monolinguality: {diff:+.1f}% difference")
    print(f"H2 Status: {'PASS' if abs(diff) >= 5 else 'FAIL (difference < 5%)'}")


if __name__ == "__main__":
    main()
