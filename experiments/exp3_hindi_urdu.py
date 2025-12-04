"""Experiment 3: Hindi-Urdu Feature Overlap (H4)

H4: Hindi and Urdu share >50% of semantic features but differ in script features

Hindi (Devanagari) and Urdu (Arabic script) are essentially the same spoken language
with different writing systems. This experiment tests whether:
- Script features are distinct (low overlap)
- Semantic features are shared (high overlap)

Falsification: Overlap <50% on parallel content
"""

import torch
import json
from pathlib import Path

from config import TARGET_LAYERS, MONOLINGUALITY_THRESHOLD, N_SAMPLES_DISCOVERY
from data import load_flores
from model import GemmaWithSAE


def compute_feature_activation_sets(model, texts, layer, threshold=0.01):
    """Get set of features that activate frequently for texts."""
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    
    activation_counts = torch.zeros(n_features, device=model.device)
    total_tokens = 0
    
    for text in texts:
        acts = model.get_sae_activations(text, layer)
        activation_counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]
    
    rates = activation_counts / total_tokens
    active_features = (rates > threshold).nonzero().squeeze(-1).tolist()
    if isinstance(active_features, int):
        active_features = [active_features]
    
    return set(active_features), rates


def compute_overlap_metrics(set1, set2):
    """Compute overlap statistics between two feature sets."""
    intersection = set1 & set2
    union = set1 | set2
    
    return {
        "jaccard": len(intersection) / len(union) if union else 0,
        "overlap_in_1": len(intersection) / len(set1) if set1 else 0,
        "overlap_in_2": len(intersection) / len(set2) if set2 else 0,
        "intersection_size": len(intersection),
        "set1_size": len(set1),
        "set2_size": len(set2),
    }


def identify_script_features(model, texts_hi, texts_ur, layer, threshold=0.05):
    """
    Identify likely script features vs semantic features.
    
    Script features: High activation for one language, low for other
    Semantic features: Similar activation for both (since content is parallel)
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    
    # Compute mean activations
    hi_acts = []
    for text in texts_hi[:100]:
        acts = model.get_sae_activations(text, layer)
        hi_acts.append(acts.mean(dim=0))
    hi_mean = torch.stack(hi_acts).mean(dim=0)
    
    ur_acts = []
    for text in texts_ur[:100]:
        acts = model.get_sae_activations(text, layer)
        ur_acts.append(acts.mean(dim=0))
    ur_mean = torch.stack(ur_acts).mean(dim=0)
    
    # Active features
    hi_active = hi_mean > threshold
    ur_active = ur_mean > threshold
    both_active = hi_active & ur_active
    
    # Features active in both with similar magnitude = semantic
    ratio = hi_mean / (ur_mean + 1e-10)
    similar_magnitude = (ratio > 0.5) & (ratio < 2.0)
    
    semantic_features = (both_active & similar_magnitude).nonzero().squeeze(-1).tolist()
    if isinstance(semantic_features, int):
        semantic_features = [semantic_features]
    
    # Features active in one but not other = script-related
    hi_only = (hi_active & ~ur_active).nonzero().squeeze(-1).tolist()
    ur_only = (ur_active & ~hi_active).nonzero().squeeze(-1).tolist()
    if isinstance(hi_only, int):
        hi_only = [hi_only]
    if isinstance(ur_only, int):
        ur_only = [ur_only]
    
    return {
        "semantic": set(semantic_features),
        "hi_script": set(hi_only),
        "ur_script": set(ur_only),
    }


def main():
    # Load model
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data
    print("Loading FLORES-200...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    texts_hi = flores["hi"]
    texts_ur = flores["ur"]
    texts_en = flores["en"]
    
    print(f"Hindi samples: {len(texts_hi)}")
    print(f"Urdu samples: {len(texts_ur)}")
    
    results = {"layers": {}}
    
    for layer in TARGET_LAYERS:
        print(f"\n=== Layer {layer} ===")
        
        # Get active feature sets
        hi_features, hi_rates = compute_feature_activation_sets(model, texts_hi, layer)
        ur_features, ur_rates = compute_feature_activation_sets(model, texts_ur, layer)
        en_features, en_rates = compute_feature_activation_sets(model, texts_en, layer)
        
        # Overall overlap
        hi_ur_overlap = compute_overlap_metrics(hi_features, ur_features)
        hi_en_overlap = compute_overlap_metrics(hi_features, en_features)
        
        print(f"Hindi features: {len(hi_features)}")
        print(f"Urdu features: {len(ur_features)}")
        print(f"Hindi-Urdu overlap: {hi_ur_overlap['jaccard']:.1%} Jaccard")
        print(f"Hindi-English overlap: {hi_en_overlap['jaccard']:.1%} Jaccard")
        
        # Script vs semantic features
        feature_types = identify_script_features(model, texts_hi, texts_ur, layer)
        
        print(f"Semantic features (shared): {len(feature_types['semantic'])}")
        print(f"Hindi script features: {len(feature_types['hi_script'])}")
        print(f"Urdu script features: {len(feature_types['ur_script'])}")
        
        results["layers"][str(layer)] = {
            "hi_features": len(hi_features),
            "ur_features": len(ur_features),
            "hi_ur_overlap": hi_ur_overlap,
            "hi_en_overlap": hi_en_overlap,
            "semantic_features": len(feature_types["semantic"]),
            "hi_script_features": len(feature_types["hi_script"]),
            "ur_script_features": len(feature_types["ur_script"]),
        }
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "exp3_hindi_urdu_overlap.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # H4 Analysis
    print("\n=== H4 Analysis ===")
    print("Hypothesis: Hindi-Urdu share >50% semantic features\n")
    
    for layer_str, data in results["layers"].items():
        semantic = data["semantic_features"]
        hi_total = data["hi_features"]
        
        overlap_ratio = semantic / hi_total if hi_total > 0 else 0
        status = "PASS" if overlap_ratio > 0.5 else "FAIL"
        
        print(f"Layer {layer_str}: {semantic}/{hi_total} semantic overlap ({overlap_ratio:.1%}) [{status}]")


if __name__ == "__main__":
    main()
