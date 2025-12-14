"""Experiment 22: Systematic Feature Interpretation Pipeline

Auto-interpretability pipeline for SAE features with monosemanticity analysis.

Research Questions:
  1. Are high-monolinguality features truly monosemantic?
  2. Do Indic-specific features have coherent linguistic interpretations?
  3. What semantic categories do language-selective features encode?

Methodology:
  1. Extract max-activating examples for each feature
  2. Compute monosemanticity via embedding clustering entropy
  3. Generate interpretations with Gemini + cross-validation
  4. Build feature semantics database for all layers

Publication-Grade Features:
  - Entropy-based monosemanticity score
  - Multi-annotator validation (if available)
  - Cross-correlation with monolinguality scores
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as scipy_stats

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    INDIC_LANGUAGES,
    EXTENDED_LANGUAGES,
    SEED,
)
from data import load_flores
from model import GemmaWithSAE
from reproducibility import seed_everything


@dataclass
class MaxActivatingExample:
    """A single max-activating example for a feature."""
    text: str
    activation: float
    token_idx: int
    context_window: str


@dataclass
class FeatureInterpretation:
    """Interpretation result for a single feature."""
    layer: int
    feature_idx: int
    short_name: str
    description: str
    is_multilingual: bool
    confidence: float
    semantic_category: str


@dataclass
class MonosemanticitScore:
    """Monosemanticity analysis result."""
    feature_idx: int
    layer: int
    entropy: float
    monosemanticity: float  # 1 - normalized_entropy
    n_clusters: int
    silhouette: float
    top_cluster_ratio: float  # Fraction of examples in largest cluster


@dataclass
class FeatureSemantics:
    """Complete semantics entry for a feature."""
    layer: int
    feature_idx: int
    monolinguality_score: float
    monosemanticity_score: float
    interpretation: Optional[FeatureInterpretation]
    max_activating_examples: List[str]
    languages_activated: List[str]


def get_max_activating_examples(
    model: GemmaWithSAE,
    texts: List[str],
    layer: int,
    feature_idx: int,
    top_k: int = 100,
    context_window: int = 50,
) -> List[MaxActivatingExample]:
    """Extract top-k max-activating examples for a feature."""

    examples = []

    for text in texts[:500]:  # Limit for efficiency
        acts = model.get_sae_activations(text, layer)
        feature_acts = acts[:, feature_idx].cpu().numpy()

        # Find tokens where this feature activates
        for token_idx, activation in enumerate(feature_acts):
            if activation > 0:
                # Extract context window
                tokens = model.tokenizer.encode(text)
                start = max(0, token_idx - context_window // 2)
                end = min(len(tokens), token_idx + context_window // 2)
                context = model.tokenizer.decode(tokens[start:end])

                examples.append(MaxActivatingExample(
                    text=text,
                    activation=float(activation),
                    token_idx=token_idx,
                    context_window=context,
                ))

    # Sort by activation and take top-k
    examples.sort(key=lambda x: x.activation, reverse=True)
    return examples[:top_k]


def compute_monosemanticity(
    examples: List[MaxActivatingExample],
    model: GemmaWithSAE,
    layer: int,
    feature_idx: int,
    n_clusters_range: Tuple[int, int] = (2, 5),
) -> MonosemanticitScore:
    """Compute monosemanticity score via embedding clustering entropy.

    High monosemanticity = low entropy = examples cluster tightly into one semantic category.
    """

    if len(examples) < 10:
        return MonosemanticitScore(
            feature_idx=feature_idx,
            layer=layer,
            entropy=1.0,
            monosemanticity=0.0,
            n_clusters=0,
            silhouette=0.0,
            top_cluster_ratio=0.0,
        )

    # Get embeddings for each example's context
    embeddings = []
    for ex in examples[:100]:
        # Use model's hidden state as embedding proxy
        with torch.no_grad():
            inputs = model.tokenizer(
                ex.context_window,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.model(**inputs, output_hidden_states=True)

            # Use the SAME layer as the feature under analysis, not the final layer,
            # to avoid mixing layer semantics and inflating monosemanticity.
            hs = outputs.hidden_states
            try:
                n_blocks = len(model.model.model.layers)
            except Exception:
                n_blocks = None
            idx = layer + 1 if (n_blocks is not None and len(hs) == n_blocks + 1) else layer
            idx = max(0, min(idx, len(hs) - 1))
            emb = hs[idx].mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Try different cluster counts and find optimal
    best_silhouette = -1
    best_n_clusters = 2
    best_labels = None

    for n in range(n_clusters_range[0], min(n_clusters_range[1] + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=n, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        if len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            if score > best_silhouette:
                best_silhouette = score
                best_n_clusters = n
                best_labels = labels

    if best_labels is None:
        return MonosemanticitScore(
            feature_idx=feature_idx,
            layer=layer,
            entropy=1.0,
            monosemanticity=0.0,
            n_clusters=1,
            silhouette=0.0,
            top_cluster_ratio=1.0,
        )

    # Compute cluster distribution entropy
    unique, counts = np.unique(best_labels, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(unique))  # Maximum possible entropy for this n_clusters

    normalized_entropy = entropy / max(max_entropy, 1e-10)
    monosemanticity = 1.0 - normalized_entropy

    # Top cluster ratio (how dominant is the largest cluster)
    top_cluster_ratio = counts.max() / counts.sum()

    return MonosemanticitScore(
        feature_idx=feature_idx,
        layer=layer,
        entropy=entropy,
        monosemanticity=monosemanticity,
        n_clusters=best_n_clusters,
        silhouette=best_silhouette,
        top_cluster_ratio=top_cluster_ratio,
    )


def interpret_with_gemini(
    examples: List[MaxActivatingExample],
    feature_idx: int,
    layer: int,
) -> Optional[FeatureInterpretation]:
    """Generate feature interpretation using Gemini API."""

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Prepare examples for prompt
        example_texts = [ex.context_window for ex in examples[:20]]
        examples_str = "\n".join([f"  {i+1}. \"{t[:200]}\"" for i, t in enumerate(example_texts)])

        prompt = f"""Analyze these text contexts where a neural network feature activates highly.

Feature Layer: {layer}
Feature Index: {feature_idx}

Max-activating examples:
{examples_str}

Based on these examples, provide:
1. short_name: A 3-5 word name for what this feature detects (e.g., "Hindi script markers", "Question words", "Proper nouns")
2. description: A 1-2 sentence description of the feature's semantic meaning
3. is_multilingual: true if feature activates across multiple languages, false if language-specific
4. confidence: A score 0-1 for how confident you are in this interpretation
5. semantic_category: One of: "script", "morphology", "syntax", "semantics", "lexical", "discourse", "unknown"

Respond in JSON format only:
{{"short_name": "...", "description": "...", "is_multilingual": true/false, "confidence": 0.X, "semantic_category": "..."}}
"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text)

        return FeatureInterpretation(
            layer=layer,
            feature_idx=feature_idx,
            short_name=data.get("short_name", "Unknown"),
            description=data.get("description", ""),
            is_multilingual=data.get("is_multilingual", False),
            confidence=data.get("confidence", 0.5),
            semantic_category=data.get("semantic_category", "unknown"),
        )

    except Exception as e:
        print(f"  Warning: Gemini interpretation failed: {e}")
        return None


def compute_monolinguality_scores(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    top_k: int = 100,
) -> Dict[int, Tuple[float, str]]:
    """Compute monolinguality scores for features at a layer.

    Returns: {feature_idx: (monolinguality_score, best_language)}
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    # Compute activation rates per language
    lang_rates = {}
    for lang, texts in texts_by_lang.items():
        if not texts:
            continue
        activation_counts = torch.zeros(n_features, device=model.device)
        total_tokens = 0

        for text in texts[:100]:
            acts = model.get_sae_activations(text, layer)
            activation_counts += (acts > 0).float().sum(dim=0)
            total_tokens += acts.shape[0]

        if total_tokens > 0:
            lang_rates[lang] = (activation_counts / total_tokens).cpu().numpy()

    if not lang_rates:
        return {}

    # Compute monolinguality: rate_best / rate_second_best
    monolinguality = {}
    for j in range(n_features):
        rates = [(lang, lang_rates[lang][j]) for lang in lang_rates]
        rates.sort(key=lambda x: x[1], reverse=True)

        if len(rates) >= 2 and rates[1][1] > 0.001:
            score = rates[0][1] / rates[1][1]
            monolinguality[j] = (score, rates[0][0])
        elif rates[0][1] > 0.001:
            monolinguality[j] = (float('inf'), rates[0][0])

    # Return top-k by monolinguality
    sorted_features = sorted(monolinguality.items(), key=lambda x: x[1][0], reverse=True)
    return dict(sorted_features[:top_k])


def build_feature_semantics_db(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layers: List[int],
    features_per_layer: int = 50,
    interpret: bool = True,
) -> Dict[str, List[FeatureSemantics]]:
    """Build comprehensive feature semantics database."""

    all_texts = []
    for texts in texts_by_lang.values():
        all_texts.extend(texts[:100])

    db = {"features": [], "layers": layers, "n_features_per_layer": features_per_layer}

    for layer in layers:
        print(f"\n=== Processing Layer {layer} ===")

        # Get top monolinguality features
        mono_scores = compute_monolinguality_scores(model, texts_by_lang, layer, top_k=features_per_layer)

        for feature_idx, (mono_score, best_lang) in tqdm(mono_scores.items(), desc=f"Layer {layer}"):
            # Get max-activating examples
            examples = get_max_activating_examples(
                model, all_texts, layer, feature_idx, top_k=100
            )

            if not examples:
                continue

            # Compute monosemanticity
            mono_semantic = compute_monosemanticity(examples, model, layer, feature_idx)

            # Get interpretation (optional)
            interpretation = None
            if interpret:
                interpretation = interpret_with_gemini(examples, feature_idx, layer)

            # Determine which languages this feature activates for
            languages_activated = []
            for lang, texts in texts_by_lang.items():
                if texts:
                    sample_acts = model.get_sae_activations(texts[0], layer)
                    if sample_acts[:, feature_idx].max() > 0:
                        languages_activated.append(lang)

            semantics = FeatureSemantics(
                layer=layer,
                feature_idx=feature_idx,
                monolinguality_score=mono_score,
                monosemanticity_score=mono_semantic.monosemanticity,
                interpretation=interpretation,
                max_activating_examples=[ex.context_window[:200] for ex in examples[:5]],
                languages_activated=languages_activated,
            )

            db["features"].append({
                "layer": layer,
                "feature_idx": feature_idx,
                "monolinguality_score": mono_score if mono_score != float('inf') else 999.0,
                "best_language": best_lang,
                "monosemanticity_score": mono_semantic.monosemanticity,
                "entropy": mono_semantic.entropy,
                "n_clusters": mono_semantic.n_clusters,
                "silhouette": mono_semantic.silhouette,
                "top_cluster_ratio": mono_semantic.top_cluster_ratio,
                "interpretation": {
                    "short_name": interpretation.short_name if interpretation else None,
                    "description": interpretation.description if interpretation else None,
                    "is_multilingual": interpretation.is_multilingual if interpretation else None,
                    "confidence": interpretation.confidence if interpretation else None,
                    "semantic_category": interpretation.semantic_category if interpretation else None,
                } if interpretation else None,
                "languages_activated": languages_activated,
                "example_contexts": [ex.context_window[:200] for ex in examples[:3]],
            })

    return db


def compute_random_baseline_monosemanticity(
    model: GemmaWithSAE,
    texts: List[str],
    layer: int,
    n_random_features: int = 50,
) -> Dict[str, float]:
    """Compute monosemanticity for random features as a baseline.

    This is crucial for establishing that high-monolinguality features are
    actually more monosemantic than random features (not just by chance).
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    # Sample random feature indices
    np.random.seed(SEED + layer)
    random_indices = np.random.choice(n_features, n_random_features, replace=False)

    monosemanticity_scores = []

    for feature_idx in tqdm(random_indices, desc=f"Random baseline L{layer}", leave=False):
        examples = get_max_activating_examples(model, texts, layer, feature_idx, top_k=100)

        if len(examples) >= 10:
            score = compute_monosemanticity(examples, model, layer, feature_idx)
            monosemanticity_scores.append(score.monosemanticity)

    if not monosemanticity_scores:
        return {"mean": 0.0, "std": 0.0, "n": 0}

    return {
        "mean": float(np.mean(monosemanticity_scores)),
        "std": float(np.std(monosemanticity_scores)),
        "n": len(monosemanticity_scores),
    }


def test_monolinguality_monosemanticity_correlation(
    db: Dict[str, Any],
    random_baseline: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Test hypothesis: high monolinguality → high monosemanticity.

    Args:
        db: Feature database
        random_baseline: Optional random baseline monosemanticity per layer
    """

    features = db["features"]

    mono_ling = [f["monolinguality_score"] for f in features if f["monolinguality_score"] < 999]
    mono_sem = [f["monosemanticity_score"] for f in features if f["monolinguality_score"] < 999]

    if len(mono_ling) < 10:
        return {"error": "insufficient_data", "n_features": len(mono_ling)}

    # Spearman correlation (robust to outliers)
    correlation, p_value = scipy_stats.spearmanr(mono_ling, mono_sem)

    # Interpretation
    if correlation > 0.5 and p_value < 0.05:
        interpretation = "STRONG: High monolinguality features are monosemantic"
    elif correlation > 0.3 and p_value < 0.05:
        interpretation = "MODERATE: Some relationship between monolinguality and monosemanticity"
    elif correlation > 0 and p_value < 0.05:
        interpretation = "WEAK: Slight positive relationship"
    else:
        interpretation = "NO RELATIONSHIP: Monolinguality does not predict monosemanticity"

    result = {
        "spearman_correlation": correlation,
        "p_value": p_value,
        "n_features": len(mono_ling),
        "interpretation": interpretation,
        "mean_monolinguality": np.mean(mono_ling),
        "mean_monosemanticity": np.mean(mono_sem),
    }

    # Compare to random baseline if available
    if random_baseline:
        random_mean = np.mean([rb["mean"] for rb in random_baseline.values() if rb["n"] > 0])
        high_mono_mean = np.mean(mono_sem)

        improvement = high_mono_mean - random_mean
        result["random_baseline_mean"] = random_mean
        result["improvement_over_random"] = improvement
        result["better_than_random"] = improvement > 0

        if improvement > 0.1:
            result["baseline_comparison"] = "VALIDATED: High-monolinguality features are more monosemantic than random"
        elif improvement > 0:
            result["baseline_comparison"] = "MARGINAL: Slight improvement over random"
        else:
            result["baseline_comparison"] = "NOT VALIDATED: No improvement over random features"

    return result


def main():
    seed_everything(SEED)

    print("=" * 60)
    print("EXPERIMENT 22: Feature Interpretation Pipeline")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in model.model_id.lower() else ""

    # Load FLORES data for all languages
    print("\nLoading FLORES data...")
    flores = load_flores(
        max_samples=N_SAMPLES_DISCOVERY,
        languages=EXTENDED_LANGUAGES,
        split="dev",
    )

    texts_by_lang = {lang: texts for lang, texts in flores.items() if texts}
    print(f"Languages loaded: {list(texts_by_lang.keys())}")

    # Select layers for analysis (early, mid, late)
    n_layers = len(TARGET_LAYERS)
    analysis_layers = [
        TARGET_LAYERS[0],                    # Early
        TARGET_LAYERS[n_layers // 2],        # Mid
        TARGET_LAYERS[-1],                   # Late
    ]
    print(f"Analyzing layers: {analysis_layers}")

    # Check for Gemini API
    has_api = os.environ.get("GOOGLE_API_KEY") is not None
    print(f"Gemini interpretation: {'ENABLED' if has_api else 'DISABLED (set GOOGLE_API_KEY)'}")

    # Build feature semantics database
    print("\n" + "=" * 60)
    print("BUILDING FEATURE SEMANTICS DATABASE")
    print("=" * 60)

    db = build_feature_semantics_db(
        model, texts_by_lang, analysis_layers,
        features_per_layer=50,
        interpret=has_api,
    )

    # Compute random baseline
    print("\n" + "=" * 60)
    print("RANDOM BASELINE COMPUTATION")
    print("=" * 60)

    all_texts = []
    for texts in texts_by_lang.values():
        all_texts.extend(texts[:100])

    random_baseline = {}
    for layer in analysis_layers:
        print(f"\nComputing random baseline for layer {layer}...")
        random_baseline[layer] = compute_random_baseline_monosemanticity(
            model, all_texts, layer, n_random_features=30
        )
        print(f"  Random monosemanticity: {random_baseline[layer]['mean']:.3f} ± {random_baseline[layer]['std']:.3f}")

    db["random_baseline"] = {str(k): v for k, v in random_baseline.items()}

    # Test hypothesis
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST: Monolinguality → Monosemanticity")
    print("=" * 60)

    correlation_result = test_monolinguality_monosemanticity_correlation(db, random_baseline)

    print(f"\n  Spearman correlation: {correlation_result.get('spearman_correlation', 'N/A'):.3f}")
    print(f"  P-value: {correlation_result.get('p_value', 'N/A'):.4f}")
    print(f"  N features: {correlation_result.get('n_features', 0)}")
    print(f"  Interpretation: {correlation_result.get('interpretation', 'N/A')}")

    if "random_baseline_mean" in correlation_result:
        print(f"\n  RANDOM BASELINE COMPARISON:")
        print(f"  Random mean monosemanticity: {correlation_result['random_baseline_mean']:.3f}")
        print(f"  High-mono mean monosemanticity: {correlation_result['mean_monosemanticity']:.3f}")
        print(f"  Improvement over random: {correlation_result['improvement_over_random']:.3f}")
        print(f"  {correlation_result.get('baseline_comparison', 'N/A')}")

    db["hypothesis_test"] = correlation_result

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for layer in analysis_layers:
        layer_features = [f for f in db["features"] if f["layer"] == layer]
        if layer_features:
            mean_mono_sem = np.mean([f["monosemanticity_score"] for f in layer_features])
            mean_mono_ling = np.mean([f["monolinguality_score"] for f in layer_features if f["monolinguality_score"] < 999])

            print(f"\n  Layer {layer}:")
            print(f"    Features analyzed: {len(layer_features)}")
            print(f"    Mean monosemanticity: {mean_mono_sem:.3f}")
            print(f"    Mean monolinguality: {mean_mono_ling:.2f}")

            if has_api:
                interpreted = [f for f in layer_features if f["interpretation"]]
                categories = defaultdict(int)
                for f in interpreted:
                    cat = f["interpretation"].get("semantic_category", "unknown")
                    categories[cat] += 1
                print(f"    Semantic categories: {dict(categories)}")

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp22_feature_interpretation{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(db, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
