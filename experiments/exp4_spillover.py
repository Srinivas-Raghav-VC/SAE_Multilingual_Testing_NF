"""Experiment 4: Language Steering Spillover Analysis

Tests: How does steering toward Hindi affect related vs unrelated languages?

Hypothesis (H4 Language Clustering):
- High spillover to Urdu (same spoken language, different script) ~50-80%
- Medium spillover to Bengali (same family: Indo-Aryan) ~20-40%  
- Low spillover to Tamil (different family: Dravidian) ~10-20%
- Minimal spillover to German/Chinese (distant) <10%

This tests whether language families cluster in SAE steering space.
"""

# Path fix for running from experiments/ directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import re

from config import (
    TARGET_LAYERS,
    STEERING_STRENGTHS,
    NUM_FEATURES,
    EVAL_PROMPTS,
    SCRIPT_RANGES,
    N_SAMPLES_EVAL,
    MIN_PROMPTS_STEERING,
)
from data import load_flores, load_research_data
from model import GemmaWithSAE


@dataclass
class SpilloverResult:
    """Results for a single steering configuration."""
    target_lang: str
    layer: int
    strength: float
    num_features: int
    language_distribution: Dict[str, float]  # Lang -> % of outputs
    n_samples: int


@dataclass  
class SpilloverAnalysis:
    """Complete spillover analysis results."""
    results_by_strength: Dict[float, SpilloverResult]
    family_analysis: Dict[str, Dict[str, float]]  # Family -> Lang -> avg spillover
    hypothesis_tests: Dict[str, bool]


def detect_script(text: str) -> str:
    """Detect primary script in text."""
    if not text:
        return "unknown"

    script_counts = {}
    for script_name, ranges in SCRIPT_RANGES.items():
        # SCRIPT_RANGES now contains lists of (start, end) tuples
        # to support multi-range scripts like Devanagari Extended
        count = 0
        for char in text:
            code = ord(char)
            for start, end in ranges:
                if start <= code <= end:
                    count += 1
                    break  # Don't double-count
        script_counts[script_name] = count

    if not script_counts or max(script_counts.values()) == 0:
        return "unknown"

    return max(script_counts, key=script_counts.get)


def script_to_language(script: str) -> str:
    """Map detected script to likely language."""
    script_lang_map = {
        "devanagari": "hi",
        "arabic": "ur_ar",  # Could be Urdu or Arabic
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "gujarati": "gu",
        "gurmukhi": "pa",
        "oriya": "or",
        "latin": "en_de_es",  # Could be English, German, Spanish, etc.
        "han": "zh",
    }
    return script_lang_map.get(script, "unknown")


def detect_language_detailed(text: str) -> str:
    """More detailed language detection using script + patterns."""
    primary_script = detect_script(text)
    
    # For Latin script, try to distinguish languages
    if primary_script == "latin":
        # Check for German-specific patterns
        german_patterns = r'\b(der|die|das|und|ist|ein|eine|ich|nicht)\b'
        spanish_patterns = r'\b(el|la|los|las|es|una|que|por|para)\b'
        french_patterns = r'\b(le|la|les|de|est|un|une|que|pour)\b'
        
        if re.search(german_patterns, text.lower()):
            return "de"
        elif re.search(spanish_patterns, text.lower()):
            return "es"
        elif re.search(french_patterns, text.lower()):
            return "fr"
        else:
            return "en"  # Default to English for Latin
    
    # For Arabic script, default to Urdu (since we're testing Hindi steering)
    if primary_script == "arabic":
        return "ur"
    
    return script_to_language(primary_script).split("_")[0]


LANGUAGE_FAMILIES = {
    "indo_aryan": ["hi", "ur", "bn", "mr", "gu", "pa", "or", "as"],
    "dravidian": ["ta", "te", "kn", "ml"],
    "germanic": ["en", "de"],
    "romance": ["es", "fr"],
    "sino_tibetan": ["zh"],
    "semitic": ["ar"],
}


def get_language_family(lang: str) -> str:
    """Get language family for a language code."""
    for family, langs in LANGUAGE_FAMILIES.items():
        if lang in langs:
            return family
    return "unknown"


def compute_steering_vector(model, texts_target, texts_source, layer):
    """Compute an EN→HI steering vector in hidden space.

    For consistency with Exp2, we:
      1. Compute activation differences in SAE feature space.
      2. Select the top-|NUM_FEATURES| features by *positive* difference
         (more active for target than source).
      3. Form a hidden-space vector as a weighted combination of the
         corresponding decoder directions, normalised to a standard
         magnitude.
    """
    sae = model.load_sae(layer)

    # Get mean activations for target language
    target_acts = []
    for text in texts_target[:100]:
        acts = model.get_sae_activations(text, layer)
        target_acts.append(acts.mean(dim=0))
    target_mean = torch.stack(target_acts).mean(dim=0)

    # Get mean activations for source language
    source_acts = []
    for text in texts_source[:100]:
        acts = model.get_sae_activations(text, layer)
        source_acts.append(acts.mean(dim=0))
    source_mean = torch.stack(source_acts).mean(dim=0)

    # Difference: positive values correspond to features more active for target.
    diff = target_mean - source_mean
    diff_clamped = torch.clamp(diff, min=0.0)

    # Select top-k features by positive difference. If, for some reason, no
    # feature is strictly more active for the target language, we fall back
    # to a dense EN→HI steering vector in hidden space. Returning a zero
    # vector here would silently disable steering and produce 100% English
    # outputs, which is exactly the suspicious pattern we want to avoid.
    if (diff_clamped > 0).sum() == 0:
        from experiments.exp2_steering import construct_dense_steering_vector

        print(
            "[exp4] Warning: no positive activation-diff features found at "
            f"layer {layer}; falling back to dense steering vector."
        )
        return construct_dense_steering_vector(model, texts_source, texts_target, layer)

    _, top_indices = diff_clamped.topk(NUM_FEATURES)

    # Weighted combination of decoder directions.
    directions = sae.W_dec[top_indices, :]  # (k, d_model)
    weights = diff_clamped[top_indices].unsqueeze(1)  # (k, 1)
    vec = (weights * directions).sum(dim=0)

    # Normalise to have comparable scale to other steering vectors.
    if vec.norm() > 0:
        vec = vec / vec.norm() * (sae.cfg.d_in ** 0.5)

    return vec


def run_spillover_experiment(
    model,
    steering_vector_hidden,
    layer,
    prompts: List[str],
    strengths: List[float],
    target_lang: str = "hi"
) -> Dict[float, SpilloverResult]:
    """Run steering at different strengths (or None for baseline), measure language distribution.
    
    Args:
        model: GemmaWithSAE model
        steering_vector_features: Steering vector in SAE FEATURE space (16384 dims)
        layer: Layer to steer
        prompts: Prompts to test
        strengths: Steering strengths
        target_lang: Target language code
    """
    results = {}
    
    # Test each strength (including 0.0 for baseline)
    test_strengths = [0.0] + list(strengths)
    
    for strength in test_strengths:
        print(f"\n  Testing strength {strength}...")
        lang_counts = defaultdict(int)
        
        for prompt in prompts:
            if strength == 0.0:
                # Baseline: no steering
                output = model.generate(prompt, max_new_tokens=64)
            else:
                # With steering (using hidden-space vector)
                output = model.generate_with_steering(
                    prompt, layer, steering_vector_hidden, strength
                )
            
            # Detect output language
            detected_lang = detect_language_detailed(output)
            lang_counts[detected_lang] += 1
        
        # Convert to percentages
        total = len(prompts)
        lang_distribution = {
            lang: count / total * 100
            for lang, count in lang_counts.items()
        }
        
        results[strength] = SpilloverResult(
            target_lang=target_lang,
            layer=layer,
            strength=strength,
            num_features=NUM_FEATURES,
            language_distribution=lang_distribution,
            n_samples=total
        )
        
        # Print summary
        print(f"    Results: {dict(lang_distribution)}")
    
    return results


def analyze_spillover_by_family(results: Dict[float, SpilloverResult]) -> Dict[str, Dict[str, float]]:
    """Analyze spillover grouped by language family."""
    analysis = {}
    
    for strength, result in results.items():
        if strength == 0.0:
            continue  # Skip baseline
        
        family_totals = defaultdict(float)
        
        for lang, percentage in result.language_distribution.items():
            family = get_language_family(lang)
            family_totals[family] += percentage
        
        analysis[str(strength)] = dict(family_totals)
    
    return analysis


def test_hypotheses(results: Dict[float, SpilloverResult]) -> Dict[str, bool]:
    """Test spillover hypotheses."""
    tests = {}
    
    # Use strength 2.0 for hypothesis tests (or highest available)
    test_strength = 2.0 if 2.0 in results else max(s for s in results.keys() if s > 0)
    result = results[test_strength]
    dist = result.language_distribution
    
    # H_spill_1: Hindi (target) should be highest
    hindi_pct = dist.get("hi", 0)
    max_other = max((v for k, v in dist.items() if k != "hi"), default=0)
    tests["hindi_is_highest"] = hindi_pct > max_other
    
    # H_spill_2: Urdu spillover > German spillover (related > unrelated)
    urdu_pct = dist.get("ur", 0)
    german_pct = dist.get("de", 0)
    tests["urdu_gt_german"] = urdu_pct > german_pct
    
    # H_spill_3: Indo-Aryan spillover > Dravidian spillover
    indo_aryan = sum(dist.get(l, 0) for l in ["hi", "ur", "bn", "mr", "gu"])
    dravidian = sum(dist.get(l, 0) for l in ["ta", "te", "kn", "ml"])
    tests["indoaryan_gt_dravidian"] = indo_aryan > dravidian
    
    # H_spill_4: Some non-target output (steering isn't perfect)
    non_hindi = 100 - hindi_pct
    tests["imperfect_steering"] = non_hindi > 5  # At least 5% spillover
    
    return tests


def main():
    """Run spillover experiment.

    We run two steering conditions:
      1. EN→HI (primary analysis for Indic clustering).
      2. EN→DE (control), showing that steering into a non-Indic language
         does NOT spuriously increase Indic outputs.

    Earlier versions of this experiment loaded the first 200 FLORES
    English sentences and used them both to construct the steering
    vectors and as evaluation prompts. This is effectively "testing on
    the training sentences" for the steering direction and risks
    overfitting to that specific sample.

    We now use ``load_research_data`` to obtain:

      - train splits (Samanantar + FLORES) for building steering
        vectors, and
      - a separate ``steering_prompts`` list derived from ``EVAL_PROMPTS``
        plus held-out FLORES EN sentences.

    This cleanly separates vector-training sentences from prompts and
    aligns Exp4 with the global train/test strategy used elsewhere.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Language Steering Spillover Analysis")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data using the unified research loader so that training
    # sentences (for steering vectors) and evaluation prompts are
    # properly separated.
    print("Loading research data for spillover experiment...")
    data_split = load_research_data()
    train_data = data_split.train
    prompts = data_split.steering_prompts

    texts_hi = train_data.get("hi", [])
    texts_de = train_data.get("de", [])
    texts_en = train_data.get("en", [])
    
    if (not texts_hi) or (not texts_en) or (not texts_de):
        print("ERROR: Could not load required train data for hi/en/de!")
        return

    if len(prompts) < MIN_PROMPTS_STEERING:
        print(
            f"[exp4] WARNING: only {len(prompts)} steering prompts available "
            f"(recommend >= {MIN_PROMPTS_STEERING} for statistically reliable spillover estimates)"
        )
    print(f"Using {len(prompts)} evaluation prompts")
    
    all_results = {"en_to_hi": {}, "en_to_de": {}}
    
    # Test at different layers for both EN→HI and EN→DE
    test_layers = [13, 20]  # Mid and late layers
    
    for layer in test_layers:
        print(f"\n{'='*20} Layer {layer} {'='*20}")
        
        # -----------------------------
        # EN → HI (primary condition)
        # -----------------------------
        print("Computing steering vector (EN → HI)...")
        steering_vector_hi = compute_steering_vector(model, texts_hi, texts_en, layer)
        
        print("Running spillover experiment for EN → HI...")
        layer_results_hi = run_spillover_experiment(
            model, steering_vector_hi, layer, prompts,
            strengths=[0.5, 1.0, 2.0, 4.0],
            target_lang="hi"
        )
        all_results["en_to_hi"][layer] = layer_results_hi
        
        print(f"\n  SPILLOVER MATRIX (EN→HI, Layer {layer}):")
        print("  " + "-" * 50)
        print(f"  {'Strength':<10} {'Hindi':<8} {'Urdu':<8} {'Bengali':<8} {'English':<8} {'German':<8}")
        print("  " + "-" * 50)
        for strength in sorted(layer_results_hi.keys()):
            result = layer_results_hi[strength]
            dist = result.language_distribution
            print(f"  {strength:<10} {dist.get('hi', 0):<8.1f} {dist.get('ur', 0):<8.1f} "
                  f"{dist.get('bn', 0):<8.1f} {dist.get('en', 0):<8.1f} {dist.get('de', 0):<8.1f}")

        # -----------------------------
        # EN → DE (control condition)
        # -----------------------------
        print("Computing steering vector (EN → DE) [control]...")
        steering_vector_de = compute_steering_vector(model, texts_de, texts_en, layer)
        
        print("Running spillover experiment for EN → DE (control)...")
        layer_results_de = run_spillover_experiment(
            model, steering_vector_de, layer, prompts,
            strengths=[0.5, 1.0, 2.0, 4.0],
            target_lang="de"
        )
        all_results["en_to_de"][layer] = layer_results_de
        
        print(f"\n  SPILLOVER MATRIX (EN→DE, Layer {layer}):")
        print("  " + "-" * 50)
        print(f"  {'Strength':<10} {'Hindi':<8} {'Urdu':<8} {'Bengali':<8} {'English':<8} {'German':<8}")
        print("  " + "-" * 50)
        for strength in sorted(layer_results_de.keys()):
            result = layer_results_de[strength]
            dist = result.language_distribution
            print(f"  {strength:<10} {dist.get('hi', 0):<8.1f} {dist.get('ur', 0):<8.1f} "
                  f"{dist.get('bn', 0):<8.1f} {dist.get('en', 0):<8.1f} {dist.get('de', 0):<8.1f}")
    
    # Hypothesis tests on best layer for EN→HI
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTS")
    print("=" * 60)
    
    # Use layer 20 for tests (typically best for steering)
    test_layer = 20 if 20 in all_results["en_to_hi"] else list(all_results["en_to_hi"].keys())[0]
    hypothesis_results = test_hypotheses(all_results["en_to_hi"][test_layer])
    
    for test_name, passed in hypothesis_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: [{status}]")
    
    overall_pass = all(hypothesis_results.values())
    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {
        "en_to_hi_layers": {
            str(layer): {
                str(strength): {
                    "language_distribution": result.language_distribution,
                    "n_samples": result.n_samples,
                }
                for strength, result in layer_results.items()
            }
            for layer, layer_results in all_results["en_to_hi"].items()
        },
        "en_to_de_layers": {
            str(layer): {
                str(strength): {
                    "language_distribution": result.language_distribution,
                    "n_samples": result.n_samples,
                }
                for strength, result in layer_results.items()
            }
            for layer, layer_results in all_results["en_to_de"].items()
        },
        "hypothesis_tests": hypothesis_results,
        "family_analysis_en_to_hi": analyze_spillover_by_family(all_results["en_to_hi"][test_layer]),
        "family_analysis_en_to_de": analyze_spillover_by_family(all_results["en_to_de"][test_layer]),
    }
    
    with open(output_dir / "exp4_spillover.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'exp4_spillover.json'}")
    
    return all_results


if __name__ == "__main__":
    main()
