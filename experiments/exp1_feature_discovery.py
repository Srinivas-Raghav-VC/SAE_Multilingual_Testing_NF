"""Experiment 1: Feature Discovery (H1, H3)

H1: SAEs contain ≥10 robust Hindi-specific features per target layer
    Falsification: <10 features with monolinguality >3.0

H3: Mid-layers (40-60% of model depth) contain most language-specific features
    For Gemma 2 2B (26 layers): 40-60% = layers 10-16
    Falsification: Peak density outside this range
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
    LANGUAGES,
    TARGET_LAYERS,
    MONOLINGUALITY_THRESHOLD,
    N_SAMPLES_DISCOVERY,
)
from data import load_flores, load_samanantar_multilingual
from model import GemmaWithSAE


def compute_activation_rates(model, texts, layer):
    """Compute per-feature activation rates across texts."""
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    
    activation_counts = torch.zeros(n_features, device=model.device)
    total_tokens = 0
    
    for text in tqdm(texts, desc=f"Layer {layer}", leave=False):
        acts = model.get_sae_activations(text, layer)  # (seq_len, n_features)
        activation_counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]
    
    return activation_counts / total_tokens  # P(feature activates)


def compute_monolinguality(rates_by_lang, target_lang):
    """
    Compute monolinguality scores for target language.
    
    M_j(L) = P(feat_j | lang L) / max_{L' != L} P(feat_j | lang L')
    """
    p_target = rates_by_lang[target_lang]
    
    other_rates = [rates_by_lang[l] for l in rates_by_lang if l != target_lang]
    p_others_max = torch.stack(other_rates).max(dim=0).values
    
    # Avoid division by zero
    mono = p_target / (p_others_max + 1e-10)
    
    # Zero out features that don't activate for target language
    mono[p_target < 1e-4] = 0
    
    return mono


def run_feature_discovery(model, texts_by_lang, layers):
    """Run feature discovery across layers."""
    results = {}
    
    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        
        # Compute activation rates per language
        rates_by_lang = {}
        for lang, texts in texts_by_lang.items():
            rates_by_lang[lang] = compute_activation_rates(model, texts, layer)
        
        # Compute monolinguality for each language
        layer_results = {"lang_features": {}, "dead_features": {}}
        
        for lang in texts_by_lang.keys():
            mono = compute_monolinguality(rates_by_lang, lang)
            
            # Find features above threshold
            specific_mask = mono > MONOLINGUALITY_THRESHOLD
            specific_ids = specific_mask.nonzero().squeeze(-1).tolist()
            if isinstance(specific_ids, int):
                specific_ids = [specific_ids]
            
            # Store with scores
            features = [(fid, mono[fid].item()) for fid in specific_ids]
            features.sort(key=lambda x: x[1], reverse=True)
            layer_results["lang_features"][lang] = features
            
            # Dead features (never activate for this language)
            dead = (rates_by_lang[lang] < 1e-6).sum().item()
            layer_results["dead_features"][lang] = dead
            
            print(f"  {lang}: {len(features)} features (M>{MONOLINGUALITY_THRESHOLD}), {dead} dead")
        
        results[layer] = layer_results
    
    return results


def analyze_h1(results, lang="hi"):
    """Analyze H1: ≥10 robust Hindi-specific features per layer."""
    print(f"\n=== H1 Analysis (M>{MONOLINGUALITY_THRESHOLD}) ===")
    
    for layer, data in sorted(results.items()):
        n_features = len(data["lang_features"].get(lang, []))
        status = "PASS" if n_features >= 10 else "FAIL"
        print(f"Layer {layer}: {n_features} {lang} features [{status}]")


def analyze_h3(results):
    """Analyze H3: Peak density in mid-layers (40-60%)."""
    print("\n=== H3 Analysis ===")
    
    # For Gemma 2 2B: 40-60% of 26 layers = layers 10-16
    mid_start, mid_end = 10, 16
    
    # Count total language-specific features per layer
    layer_counts = {}
    for layer, data in results.items():
        total = sum(len(feats) for feats in data["lang_features"].values())
        layer_counts[layer] = total
    
    # Find peak
    peak_layer = max(layer_counts, key=layer_counts.get)
    in_mid = mid_start <= peak_layer <= mid_end
    
    print(f"Feature counts by layer: {layer_counts}")
    print(f"Peak layer: {peak_layer} (mid-range: {mid_start}-{mid_end})")
    print(f"H3 Status: {'PASS' if in_mid else 'FAIL'}")


def main():
    # Load model
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data for feature discovery
    # Use Samanantar for high-resource Indic languages where available,
    # and FLORES for the remaining languages (including controls).
    print("Loading data for feature discovery...")
    
    # Indic languages covered by Samanantar and used in this experiment
    samanantar_langs = ["hi", "bn", "ta", "te"]
    texts_by_lang = {}
    
    print("  Loading Samanantar for Indic languages (hi, bn, ta, te)...")
    sam_data = load_samanantar_multilingual(
        samanantar_langs, max_samples_per_lang=N_SAMPLES_DISCOVERY
    )
    
    # Fill Indic languages from Samanantar
    for lang in samanantar_langs:
        if lang in sam_data and sam_data[lang]:
            texts_by_lang[lang] = sam_data[lang]
    
    # Ensure we have English; prefer Samanantar English if present
    if "en" in sam_data and sam_data["en"]:
        texts_by_lang["en"] = sam_data["en"]
    
    # Load remaining languages from FLORES (including Urdu + controls)
    flores_langs = {}
    for lang, flores_code in LANGUAGES.items():
        if lang in texts_by_lang:
            continue
        flores_langs[lang] = flores_code
    
    if flores_langs:
        print("  Loading FLORES-200 for remaining languages...")
        flores = load_flores(max_samples=N_SAMPLES_DISCOVERY, languages=flores_langs)
        for lang, sents in flores.items():
            texts_by_lang[lang] = sents
    
    # Sanity check: ensure we have at least the core languages
    missing = [lang for lang in LANGUAGES.keys() if lang not in texts_by_lang]
    if missing:
        print(f"WARNING: Missing languages in feature discovery data: {missing}")
    
    # Log sample sizes
    print("Sample counts per language for feature discovery:")
    for lang in sorted(texts_by_lang.keys()):
        print(f"  {lang}: {len(texts_by_lang[lang])} sentences")
    
    # Run discovery
    results = run_feature_discovery(model, texts_by_lang, TARGET_LAYERS)
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Convert tensors for JSON serialization
    json_results = {}
    for layer, data in results.items():
        json_results[str(layer)] = {
            "lang_features": data["lang_features"],
            "dead_features": data["dead_features"],
        }
    
    with open(output_dir / "exp1_feature_discovery.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'exp1_feature_discovery.json'}")
    
    # Analyze hypotheses
    analyze_h1(results)
    analyze_h3(results)


if __name__ == "__main__":
    main()
