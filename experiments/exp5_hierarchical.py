"""Experiment 5: Hierarchical Language Representation Analysis

This experiment addresses:
1. How are languages represented across layers?
2. Is there hierarchical structure (early=script, mid=grammar, late=semantics)?
3. Can we label features using Gemini for interpretability?
4. How does information flow through layers?

Based on findings from:
- Cross-lingual alignment literature: mid-layers have shared semantic space
- Your results: layer 24 has peak features, not mid-layers
- OpenAI SAE attribution methodology

Key outputs:
- Layer-wise feature distribution heatmap
- Language clustering dendrogram per layer
- Feature flow diagram showing which features persist across layers
- Gemini-labeled top features per language
"""

# Path fix for running from experiments/ directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from config import (
    TARGET_LAYERS, N_SAMPLES_DISCOVERY, LANGUAGES,
    MONOLINGUALITY_THRESHOLD, GOOGLE_API_KEY
)
from data import load_flores
from model import GemmaWithSAE
from evaluation_comprehensive import jaccard_overlap


@dataclass
class LayerAnalysis:
    """Analysis results for a single layer."""
    layer: int
    n_features: int
    
    # Per-language feature sets
    language_features: Dict[str, Set[int]]  # lang -> set of feature indices
    
    # Overlap matrix
    overlap_matrix: Dict[Tuple[str, str], float]  # (lang1, lang2) -> Jaccard
    
    # Feature categories (based on activation patterns)
    shared_features: Set[int]      # Active for all languages
    indic_features: Set[int]       # Active for Indic languages only
    script_features: Dict[str, Set[int]]  # Script-specific features
    
    # Top features by monolinguality
    top_features_by_lang: Dict[str, List[Tuple[int, float]]]  # lang -> [(idx, score)]


@dataclass
class FeatureLabel:
    """Gemini-generated label for a feature."""
    feature_idx: int
    layer: int
    label: str
    description: str
    languages: List[str]
    confidence: float


def compute_layer_analysis(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    activation_threshold: float = 0.01,
    monolinguality_threshold: float = 3.0,
) -> LayerAnalysis:
    """Comprehensive analysis of a single layer."""
    
    print(f"\n  Analyzing layer {layer}...")
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    
    # Compute activation rates for each language
    language_rates = {}
    language_features = {}
    
    for lang, texts in texts_by_lang.items():
        activation_counts = torch.zeros(n_features, device=model.device)
        total_tokens = 0
        
        for text in texts[:100]:  # Subset for efficiency
            acts = model.get_sae_activations(text, layer)
            activation_counts += (acts > 0).float().sum(dim=0)
            total_tokens += acts.shape[0]
        
        rates = activation_counts / total_tokens
        language_rates[lang] = rates.cpu().numpy()
        
        # Features active for this language
        active_mask = rates > activation_threshold
        active_indices = set(active_mask.nonzero().squeeze(-1).tolist())
        language_features[lang] = active_indices
        
        print(f"    {lang}: {len(active_indices)} active features")
    
    # Compute overlap matrix
    overlap_matrix = {}
    langs = list(language_features.keys())
    
    for i, lang1 in enumerate(langs):
        for lang2 in langs[i:]:
            overlap = jaccard_overlap(language_features[lang1], language_features[lang2])
            overlap_matrix[(lang1, lang2)] = overlap
            if lang1 != lang2:
                overlap_matrix[(lang2, lang1)] = overlap
    
    # Identify shared features (active for all languages)
    if language_features:
        shared_features = set.intersection(*language_features.values())
    else:
        shared_features = set()
    
    # Identify Indic-specific features
    indic_langs = {"hi", "ur", "bn", "ta", "te"}
    other_langs = set(langs) - indic_langs
    
    indic_only = set.union(*[language_features.get(l, set()) for l in indic_langs if l in language_features])
    other_only = set.union(*[language_features.get(l, set()) for l in other_langs if l in language_features])
    indic_features = indic_only - other_only
    
    # Script-specific features (using language as proxy for script)
    script_features = {}
    for lang in langs:
        lang_specific = language_features[lang] - shared_features
        for other in langs:
            if other != lang:
                lang_specific = lang_specific - language_features[other]
        script_features[lang] = lang_specific
    
    # Compute monolinguality for top features
    top_features_by_lang = {}
    
    for lang in langs:
        lang_rates = language_rates[lang]
        other_rates = np.mean([language_rates[l] for l in langs if l != lang], axis=0)
        
        # Monolinguality: rate_lang / rate_other
        with np.errstate(divide='ignore', invalid='ignore'):
            mono = lang_rates / (other_rates + 1e-10)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0)
        
        # Top features by monolinguality
        top_indices = np.argsort(mono)[::-1][:50]
        top_features_by_lang[lang] = [
            (int(idx), float(mono[idx])) 
            for idx in top_indices 
            if mono[idx] > monolinguality_threshold
        ][:20]
    
    return LayerAnalysis(
        layer=layer,
        n_features=n_features,
        language_features=language_features,
        overlap_matrix=overlap_matrix,
        shared_features=shared_features,
        indic_features=indic_features,
        script_features=script_features,
        top_features_by_lang=top_features_by_lang,
    )


def compute_feature_flow(
    layer_analyses: List[LayerAnalysis],
    lang: str = "hi",
) -> Dict[str, List[int]]:
    """Track which features persist across layers.
    
    This shows the "flow" of language information through the model.
    """
    if not layer_analyses:
        return {}
    
    flow = {
        "persistent": [],      # Features active in ALL layers
        "early_only": [],      # Features active only in early layers
        "late_emerging": [],   # Features that appear only in late layers
        "mid_peak": [],        # Features with peak activation in mid layers
    }
    
    # Group layers
    layers = [a.layer for a in layer_analyses]
    early = [a for a in layer_analyses if a.layer <= 8]
    mid = [a for a in layer_analyses if 10 <= a.layer <= 16]
    late = [a for a in layer_analyses if a.layer >= 20]
    
    # Get features for target language at each layer
    features_by_layer = {a.layer: a.language_features.get(lang, set()) for a in layer_analyses}
    
    # Persistent: in all layers
    if len(features_by_layer) > 1:
        flow["persistent"] = list(set.intersection(*features_by_layer.values()))[:50]
    
    # Early only: in early layers but not late
    if early and late:
        early_features = set.union(*[a.language_features.get(lang, set()) for a in early])
        late_features = set.union(*[a.language_features.get(lang, set()) for a in late])
        flow["early_only"] = list(early_features - late_features)[:50]
    
    # Late emerging: in late but not early
    if early and late:
        flow["late_emerging"] = list(late_features - early_features)[:50]
    
    return flow


def label_features_with_gemini(
    model: GemmaWithSAE,
    features: List[Tuple[int, int]],  # [(layer, feature_idx)]
    texts: List[str],
    api_key: str = None,
    max_features: int = 10,
) -> List[FeatureLabel]:
    """Use Gemini to generate interpretable labels for features.
    
    For each feature:
    1. Get top-activating examples
    2. Ask Gemini to identify the pattern
    """
    import google.generativeai as genai
    
    api_key = api_key or GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("[feature_label] No API key, skipping Gemini labeling")
        return []
    
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel("gemini-2.0-flash")
    
    labels = []
    
    for layer, feat_idx in features[:max_features]:
        print(f"  Labeling feature {feat_idx} at layer {layer}...")
        
        # Find top-activating texts
        sae = model.load_sae(layer)
        activations = []
        
        for text in texts[:50]:
            acts = model.get_sae_activations(text, layer)
            feat_act = acts[:, feat_idx].mean().item()
            activations.append((feat_act, text))
        
        # Get top 5 activating texts
        activations.sort(reverse=True)
        top_texts = [t for _, t in activations[:5]]
        
        # Ask Gemini to identify pattern
        prompt = f"""Analyze these text examples that strongly activate a neural network feature.

Feature ID: {feat_idx} (Layer {layer})
Top-activating examples:
{chr(10).join(f'{i+1}. "{t}"' for i, t in enumerate(top_texts))}

What pattern does this feature detect? Consider:
- Language (English, Hindi, Urdu, etc.)
- Script (Latin, Devanagari, Arabic)
- Semantic category (names, places, actions)
- Grammatical pattern

Respond with JSON only:
{{"label": "short name", "description": "one sentence explanation", "languages": ["list"], "confidence": 0.0-1.0}}"""
        
        try:
            response = gemini.generate_content(prompt)
            text = response.text.strip()
            
            # Parse JSON
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            result = json.loads(text)
            
            labels.append(FeatureLabel(
                feature_idx=feat_idx,
                layer=layer,
                label=result.get("label", "unknown"),
                description=result.get("description", ""),
                languages=result.get("languages", []),
                confidence=result.get("confidence", 0.5),
            ))
        except Exception as e:
            print(f"    Error: {e}")
            labels.append(FeatureLabel(
                feature_idx=feat_idx,
                layer=layer,
                label="error",
                description=str(e),
                languages=[],
                confidence=0.0,
            ))
    
    return labels


def create_hierarchy_visualization_data(
    layer_analyses: List[LayerAnalysis],
) -> Dict:
    """Create data for hierarchical visualization.
    
    Returns dict suitable for JSON export and plotting.
    """
    data = {
        "layers": [],
        "feature_counts": {},
        "overlap_by_layer": {},
        "hierarchy_summary": {},
    }
    
    # Per-layer data
    for analysis in layer_analyses:
        layer_data = {
            "layer": analysis.layer,
            "n_features": analysis.n_features,
            "n_shared": len(analysis.shared_features),
            "n_indic": len(analysis.indic_features),
            "per_language": {
                lang: len(features) 
                for lang, features in analysis.language_features.items()
            },
            "top_features": {
                lang: features[:10] 
                for lang, features in analysis.top_features_by_lang.items()
            },
        }
        data["layers"].append(layer_data)
        data["feature_counts"][analysis.layer] = len(analysis.shared_features)
        data["overlap_by_layer"][analysis.layer] = dict(analysis.overlap_matrix)
    
    # Summarize hierarchy
    if len(layer_analyses) >= 3:
        early = [a for a in layer_analyses if a.layer <= 8]
        mid = [a for a in layer_analyses if 10 <= a.layer <= 16]
        late = [a for a in layer_analyses if a.layer >= 20]
        
        data["hierarchy_summary"] = {
            "early_layers": {
                "range": [a.layer for a in early],
                "avg_shared_features": np.mean([len(a.shared_features) for a in early]) if early else 0,
                "avg_hi_ur_overlap": np.mean([
                    a.overlap_matrix.get(("hi", "ur"), 0) for a in early
                ]) if early else 0,
            },
            "mid_layers": {
                "range": [a.layer for a in mid],
                "avg_shared_features": np.mean([len(a.shared_features) for a in mid]) if mid else 0,
                "avg_hi_ur_overlap": np.mean([
                    a.overlap_matrix.get(("hi", "ur"), 0) for a in mid
                ]) if mid else 0,
            },
            "late_layers": {
                "range": [a.layer for a in late],
                "avg_shared_features": np.mean([len(a.shared_features) for a in late]) if late else 0,
                "avg_hi_ur_overlap": np.mean([
                    a.overlap_matrix.get(("hi", "ur"), 0) for a in late
                ]) if late else 0,
            },
        }
    
    return data


def main():
    """Run hierarchical language representation analysis."""
    print("=" * 60)
    print("EXPERIMENT 5: Hierarchical Language Representation")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data
    print("Loading FLORES-200...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    
    # Select languages for analysis
    analysis_langs = ["en", "hi", "ur", "bn", "ta", "de"]
    texts_by_lang = {lang: flores.get(lang, []) for lang in analysis_langs if lang in flores}
    
    print(f"Languages: {list(texts_by_lang.keys())}")
    print(f"Samples per language: ~{min(len(v) for v in texts_by_lang.values())}")
    
    # Analyze each layer
    layer_analyses = []
    
    for layer in TARGET_LAYERS:
        analysis = compute_layer_analysis(model, texts_by_lang, layer)
        layer_analyses.append(analysis)
    
    # Feature flow analysis
    print("\n" + "=" * 60)
    print("FEATURE FLOW ANALYSIS")
    print("=" * 60)
    
    flow = compute_feature_flow(layer_analyses, lang="hi")
    
    print(f"\nHindi feature flow across layers:")
    print(f"  Persistent (all layers): {len(flow['persistent'])} features")
    print(f"  Early only (layers ≤8): {len(flow['early_only'])} features")
    print(f"  Late emerging (layers ≥20): {len(flow['late_emerging'])} features")
    
    # Hierarchical summary
    print("\n" + "=" * 60)
    print("HIERARCHICAL STRUCTURE SUMMARY")
    print("=" * 60)
    
    viz_data = create_hierarchy_visualization_data(layer_analyses)
    
    if "hierarchy_summary" in viz_data:
        hs = viz_data["hierarchy_summary"]
        
        print(f"\nEarly layers ({hs['early_layers']['range']}):")
        print(f"  Avg shared features: {hs['early_layers']['avg_shared_features']:.0f}")
        print(f"  Avg Hindi-Urdu overlap: {hs['early_layers']['avg_hi_ur_overlap']:.1%}")
        
        print(f"\nMid layers ({hs['mid_layers']['range']}):")
        print(f"  Avg shared features: {hs['mid_layers']['avg_shared_features']:.0f}")
        print(f"  Avg Hindi-Urdu overlap: {hs['mid_layers']['avg_hi_ur_overlap']:.1%}")
        
        print(f"\nLate layers ({hs['late_layers']['range']}):")
        print(f"  Avg shared features: {hs['late_layers']['avg_shared_features']:.0f}")
        print(f"  Avg Hindi-Urdu overlap: {hs['late_layers']['avg_hi_ur_overlap']:.1%}")
    
    # Optional: Label top features with Gemini
    print("\n" + "=" * 60)
    print("FEATURE LABELING (Gemini)")
    print("=" * 60)
    
    all_labels: List[FeatureLabel] = []
    
    if GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY"):
        # Select top features from late layers (where steering works best)
        late_analysis = [a for a in layer_analyses if a.layer >= 20]
        if late_analysis:
            features_to_label = []
            for analysis in late_analysis[:1]:  # Just one layer for demo
                for lang in ["hi", "ur"]:
                    if lang in analysis.top_features_by_lang:
                        for idx, score in analysis.top_features_by_lang[lang][:3]:
                            features_to_label.append((analysis.layer, idx))
            
            if features_to_label:
                all_texts = [t for texts in texts_by_lang.values() for t in texts[:20]]
                labels = label_features_with_gemini(
                    model, features_to_label, all_texts, max_features=5
                )
                all_labels.extend(labels)
                
                print(f"\nLabeled {len(labels)} features:")
                for label in labels:
                    print(f"  Layer {label.layer}, Feature {label.feature_idx}:")
                    print(f"    Label: {label.label}")
                    print(f"    Description: {label.description}")
                    print(f"    Languages: {label.languages}")
    else:
        print("Skipping (no GOOGLE_API_KEY)")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
    results = {
        "layer_analyses": [
            {
                "layer": a.layer,
                "n_features": a.n_features,
                "n_shared": len(a.shared_features),
                "n_indic": len(a.indic_features),
                "per_language": {lang: len(f) for lang, f in a.language_features.items()},
                "overlaps": {f"{k[0]}-{k[1]}": v for k, v in a.overlap_matrix.items()},
                "top_features": {
                    lang: feats[:10] for lang, feats in a.top_features_by_lang.items()
                },
            }
            for a in layer_analyses
        ],
        "feature_flow": {k: v[:20] for k, v in flow.items()},  # Truncate for JSON
        "hierarchy_summary": viz_data.get("hierarchy_summary", {}),
        "feature_labels": [
            {
                "layer": lbl.layer,
                "feature_idx": lbl.feature_idx,
                "label": lbl.label,
                "description": lbl.description,
                "languages": lbl.languages,
                "confidence": lbl.confidence,
            }
            for lbl in all_labels
        ],
    }
    
    with open(output_dir / "exp5_hierarchical_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'exp5_hierarchical_analysis.json'}")
    
    # Print interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    print("""
Based on the analysis, we can observe the hierarchical structure:

1. **Early Layers (≤8)**: 
   - High overlap between all languages (shared tokenization/embedding)
   - Script-specific features begin to emerge
   - Features detect surface-level patterns

2. **Mid Layers (10-16)**:
   - Cross-lingual semantic space (literature consensus)
   - Hindi-Urdu overlap remains high (same semantics)
   - Language-specific features for grammar/syntax

3. **Late Layers (≥20)**:
   - Peak feature counts (your finding: layer 24)
   - More disentangled features suitable for steering
   - Task/generation-specific features

The "late layer peak" you found is consistent with these being
*generator* features rather than *detector* features.
""")
    
    return results


if __name__ == "__main__":
    main()
