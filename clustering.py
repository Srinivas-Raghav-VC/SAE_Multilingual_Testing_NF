"""Language Clustering Analysis for SAE Features.

This module analyzes how languages cluster in SAE feature space, testing:
- H4: Do language families cluster together?
- H5: How do steering spillover effects propagate?

Key analyses:
1. Feature overlap (Jaccard similarity) between language pairs
2. Hierarchical clustering of languages
3. Steering spillover measurement
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, clustering will be limited")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LanguageFeatures:
    """Features for a single language."""
    code: str
    name: str
    family: str
    script: str
    active_features: Set[int]          # Features that activate for this language
    feature_activations: Dict[int, float]  # Feature ID -> mean activation
    
    def overlap_with(self, other: "LanguageFeatures") -> float:
        """Compute Jaccard overlap with another language."""
        intersection = len(self.active_features & other.active_features)
        union = len(self.active_features | other.active_features)
        if union == 0:
            return 0.0
        return intersection / union


@dataclass
class ClusteringResults:
    """Results of language clustering analysis."""
    overlap_matrix: Dict[Tuple[str, str], float]
    family_overlaps: Dict[str, float]
    script_overlaps: Dict[str, float]
    dendrogram_data: Optional[Dict] = None
    cluster_assignments: Optional[Dict[str, int]] = None


@dataclass
class SpilloverResults:
    """Results of steering spillover analysis."""
    target_lang: str
    steering_strength: float
    language_effects: Dict[str, float]  # Lang -> % output in that language
    family_effects: Dict[str, float]    # Family -> average effect


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_language_features(
    model,
    texts: List[str],
    layer: int,
    activation_threshold: float = 0.1
) -> LanguageFeatures:
    """Extract active SAE features for a language.
    
    Args:
        model: GemmaWithSAE model
        texts: List of texts in the language
        layer: Layer to analyze
        activation_threshold: Threshold for "active" feature
        
    Returns:
        LanguageFeatures object
    """
    from tqdm import tqdm
    
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    
    # Track activations
    feature_sums = torch.zeros(n_features, device=model.device)
    feature_counts = torch.zeros(n_features, device=model.device)
    total_tokens = 0
    
    for text in tqdm(texts, desc="Extracting features", leave=False):
        acts = model.get_sae_activations(text, layer)  # (seq_len, n_features)
        
        # Sum activations
        feature_sums += acts.sum(dim=0)
        feature_counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]
    
    # Mean activation per feature
    mean_acts = feature_sums / max(total_tokens, 1)
    
    # Active features (those that activate above threshold on average)
    active_mask = mean_acts > activation_threshold
    active_features = set(active_mask.nonzero().squeeze(-1).tolist())
    
    # Feature activations dict
    feature_activations = {
        i: mean_acts[i].item() 
        for i in range(n_features) 
        if mean_acts[i] > 0
    }
    
    return active_features, feature_activations


def compute_language_features_batch(
    model,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    activation_threshold: float = 0.1,
    language_metadata: Optional[Dict] = None
) -> Dict[str, LanguageFeatures]:
    """Compute features for multiple languages.
    
    Args:
        model: GemmaWithSAE model
        texts_by_lang: Dict mapping language code to texts
        layer: Layer to analyze
        activation_threshold: Threshold for active features
        language_metadata: Optional metadata for languages
        
    Returns:
        Dict mapping language code to LanguageFeatures
    """
    if language_metadata is None:
        from data import LANGUAGE_METADATA
        language_metadata = LANGUAGE_METADATA
    
    results = {}
    
    for lang, texts in texts_by_lang.items():
        print(f"\nProcessing {lang}...")
        
        active_features, feature_activations = extract_language_features(
            model, texts, layer, activation_threshold
        )
        
        meta = language_metadata.get(lang, {
            "name": lang,
            "family": "Unknown",
            "script": "Unknown"
        })
        
        results[lang] = LanguageFeatures(
            code=lang,
            name=meta.get("name", lang),
            family=meta.get("family", "Unknown"),
            script=meta.get("script", "Unknown"),
            active_features=active_features,
            feature_activations=feature_activations
        )
        
        print(f"  {lang}: {len(active_features)} active features")
    
    return results


# =============================================================================
# OVERLAP ANALYSIS (CORRECT JACCARD!)
# =============================================================================

def compute_jaccard_overlap(set_a: Set[int], set_b: Set[int]) -> float:
    """Compute Jaccard overlap between two sets.
    
    CORRECT FORMULA: |A ∩ B| / |A ∪ B|
    This is ALWAYS between 0 and 1.
    
    Args:
        set_a: First set
        set_b: Second set
        
    Returns:
        Jaccard similarity (0 to 1)
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    jaccard = intersection / union
    
    # Sanity check (should never fail)
    assert 0.0 <= jaccard <= 1.0, f"Invalid Jaccard: {jaccard}"
    
    return jaccard


def compute_overlap_matrix(
    language_features: Dict[str, LanguageFeatures]
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise overlap matrix.
    
    Args:
        language_features: Dict of LanguageFeatures
        
    Returns:
        Dict mapping (lang1, lang2) to Jaccard overlap
    """
    languages = list(language_features.keys())
    overlap = {}
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i <= j:  # Include diagonal
                feat1 = language_features[lang1].active_features
                feat2 = language_features[lang2].active_features
                
                jaccard = compute_jaccard_overlap(feat1, feat2)
                overlap[(lang1, lang2)] = jaccard
                overlap[(lang2, lang1)] = jaccard  # Symmetric
    
    return overlap


def compute_family_overlaps(
    language_features: Dict[str, LanguageFeatures]
) -> Dict[str, float]:
    """Compute average overlap within vs between families.
    
    Args:
        language_features: Dict of LanguageFeatures
        
    Returns:
        Dict with "within_family" and "between_family" averages
    """
    within_overlaps = []
    between_overlaps = []
    
    languages = list(language_features.keys())
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i < j:  # Don't include diagonal or duplicates
                feat1 = language_features[lang1]
                feat2 = language_features[lang2]
                
                jaccard = compute_jaccard_overlap(
                    feat1.active_features, 
                    feat2.active_features
                )
                
                if feat1.family == feat2.family:
                    within_overlaps.append(jaccard)
                else:
                    between_overlaps.append(jaccard)
    
    return {
        "within_family": np.mean(within_overlaps) if within_overlaps else 0.0,
        "between_family": np.mean(between_overlaps) if between_overlaps else 0.0,
        "ratio": (np.mean(within_overlaps) / np.mean(between_overlaps) 
                  if between_overlaps and np.mean(between_overlaps) > 0 else 0.0)
    }


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def cluster_languages(
    language_features: Dict[str, LanguageFeatures],
    method: str = "average"
) -> Optional[Dict]:
    """Perform hierarchical clustering of languages.
    
    Args:
        language_features: Dict of LanguageFeatures
        method: Linkage method ("average", "complete", "single")
        
    Returns:
        Dict with clustering results or None if scipy unavailable
    """
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available for clustering")
        return None
    
    languages = list(language_features.keys())
    n_langs = len(languages)
    
    # Build distance matrix (1 - Jaccard similarity)
    dist_matrix = np.zeros((n_langs, n_langs))
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i != j:
                jaccard = compute_jaccard_overlap(
                    language_features[lang1].active_features,
                    language_features[lang2].active_features
                )
                dist_matrix[i, j] = 1 - jaccard
    
    # Perform hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=method)
    
    return {
        "languages": languages,
        "linkage_matrix": Z.tolist(),
        "distance_matrix": dist_matrix.tolist(),
        "method": method
    }


# =============================================================================
# STEERING SPILLOVER ANALYSIS
# =============================================================================

def measure_steering_spillover(
    model,
    steering_vector: torch.Tensor,
    layer: int,
    prompts: List[str],
    strengths: List[float],
    target_lang: str = "hi"
) -> List[SpilloverResults]:
    """Measure how steering toward one language affects others.
    
    Args:
        model: GemmaWithSAE model
        steering_vector: Vector to add during generation
        layer: Layer to apply steering
        prompts: Prompts to test
        strengths: Steering strengths to test
        target_lang: Target language for steering
        
    Returns:
        List of SpilloverResults for each strength
    """
    from evaluation import detect_scripts
    
    results = []
    
    for strength in strengths:
        print(f"\nTesting strength {strength}...")
        
        # Track language outputs
        lang_counts = defaultdict(int)
        total = 0
        
        for prompt in prompts:
            output = model.generate_with_steering(
                prompt, layer, steering_vector, strength
            )
            
            # Detect primary script/language
            scripts = detect_scripts(output)
            
            # Assign to most likely language
            if scripts.get("devanagari", 0) > 0.3:
                lang_counts["hi"] += 1
            elif scripts.get("arabic", 0) > 0.3:
                lang_counts["ur"] += 1  # or Arabic
            elif scripts.get("bengali", 0) > 0.3:
                lang_counts["bn"] += 1
            elif scripts.get("tamil", 0) > 0.3:
                lang_counts["ta"] += 1
            elif scripts.get("telugu", 0) > 0.3:
                lang_counts["te"] += 1
            else:
                lang_counts["other"] += 1
            
            total += 1
        
        # Convert to percentages
        lang_effects = {
            lang: count / total * 100
            for lang, count in lang_counts.items()
        }
        
        results.append(SpilloverResults(
            target_lang=target_lang,
            steering_strength=strength,
            language_effects=lang_effects,
            family_effects={}  # TODO: Compute family-level effects
        ))
    
    return results


# =============================================================================
# SCRIPT VS SEMANTIC ANALYSIS
# =============================================================================

def analyze_script_vs_semantic(
    lang1_features: LanguageFeatures,
    lang2_features: LanguageFeatures
) -> Dict:
    """Analyze script-specific vs semantic features between two languages.
    
    Particularly useful for Hindi-Urdu comparison (same language, different scripts).
    
    Args:
        lang1_features: Features for first language (e.g., Hindi)
        lang2_features: Features for second language (e.g., Urdu)
        
    Returns:
        Dict with semantic (shared) and script-specific features
    """
    feat1 = lang1_features.active_features
    feat2 = lang2_features.active_features
    
    # Shared = semantic (activate for both)
    shared = feat1 & feat2
    
    # Unique = script-specific
    only_1 = feat1 - feat2
    only_2 = feat2 - feat1
    
    return {
        "lang1": lang1_features.code,
        "lang2": lang2_features.code,
        "semantic_features": len(shared),
        "script_features_lang1": len(only_1),
        "script_features_lang2": len(only_2),
        "semantic_ratio": len(shared) / len(feat1 | feat2) if feat1 | feat2 else 0,
        "script_ratio_lang1": len(only_1) / len(feat1) if feat1 else 0,
        "script_ratio_lang2": len(only_2) / len(feat2) if feat2 else 0,
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_clustering_analysis(
    model,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    output_dir: str = "results",
    activation_threshold: float = 0.1
) -> ClusteringResults:
    """Run complete clustering analysis.
    
    Args:
        model: GemmaWithSAE model
        texts_by_lang: Dict mapping language to texts
        layer: Layer to analyze
        output_dir: Directory to save results
        activation_threshold: Threshold for active features
        
    Returns:
        ClusteringResults object
    """
    print("\n" + "=" * 60)
    print(f"CLUSTERING ANALYSIS - Layer {layer}")
    print("=" * 60)
    
    # 1. Extract features for each language
    print("\n1. Extracting language features...")
    language_features = compute_language_features_batch(
        model, texts_by_lang, layer, activation_threshold
    )
    
    # 2. Compute overlap matrix
    print("\n2. Computing overlap matrix...")
    overlap_matrix = compute_overlap_matrix(language_features)
    
    # Print overlap matrix
    languages = list(language_features.keys())
    print("\nJaccard Overlap Matrix:")
    print("     " + "  ".join(f"{l:>5}" for l in languages))
    for l1 in languages:
        row = "  ".join(f"{overlap_matrix[(l1, l2)]:5.2f}" for l2 in languages)
        print(f"{l1:>5} {row}")
    
    # 3. Compute family overlaps
    print("\n3. Computing family overlaps...")
    family_overlaps = compute_family_overlaps(language_features)
    print(f"  Within-family average: {family_overlaps['within_family']:.3f}")
    print(f"  Between-family average: {family_overlaps['between_family']:.3f}")
    print(f"  Ratio: {family_overlaps['ratio']:.2f}x")
    
    # 4. Hierarchical clustering
    print("\n4. Performing hierarchical clustering...")
    dendrogram_data = cluster_languages(language_features)
    
    # 5. Script vs Semantic analysis (Hindi-Urdu if available)
    if "hi" in language_features and "ur" in language_features:
        print("\n5. Hindi-Urdu Script vs Semantic analysis...")
        hi_ur_analysis = analyze_script_vs_semantic(
            language_features["hi"],
            language_features["ur"]
        )
        print(f"  Semantic (shared) features: {hi_ur_analysis['semantic_features']}")
        print(f"  Hindi-only (script) features: {hi_ur_analysis['script_features_lang1']}")
        print(f"  Urdu-only (script) features: {hi_ur_analysis['script_features_lang2']}")
        print(f"  Semantic ratio: {hi_ur_analysis['semantic_ratio']:.1%}")
    
    # 6. Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = ClusteringResults(
        overlap_matrix=overlap_matrix,
        family_overlaps=family_overlaps,
        script_overlaps={},  # TODO
        dendrogram_data=dendrogram_data
    )
    
    # Save to JSON
    with open(output_path / f"clustering_layer_{layer}.json", "w") as f:
        json.dump({
            "layer": layer,
            "overlap_matrix": {f"{k[0]}-{k[1]}": v for k, v in overlap_matrix.items()},
            "family_overlaps": family_overlaps,
            "dendrogram_data": dendrogram_data,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path / f'clustering_layer_{layer}.json'}")
    
    return results


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing clustering module...")
    
    # Test Jaccard computation
    set_a = {1, 2, 3, 4, 5}
    set_b = {3, 4, 5, 6, 7}
    
    jaccard = compute_jaccard_overlap(set_a, set_b)
    expected = 3 / 7  # Intersection = {3,4,5} = 3, Union = {1,2,3,4,5,6,7} = 7
    
    print(f"Jaccard test: {jaccard:.4f} (expected {expected:.4f})")
    assert abs(jaccard - expected) < 0.001, "Jaccard computation failed!"
    
    print("✓ All tests passed!")
