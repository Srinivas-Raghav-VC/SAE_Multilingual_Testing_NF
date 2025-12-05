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
    TARGET_LAYERS, STEERING_STRENGTHS, NUM_FEATURES,
    EVAL_PROMPTS, SCRIPT_RANGES
)
from data import load_flores
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
    for script_name, (start, end) in SCRIPT_RANGES.items():
        count = sum(1 for c in text if start <= ord(c) <= end)
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
    """Compute activation-diff steering vector."""
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
    
    # Steering direction
    diff = target_mean - source_mean
    
    # Select top-k features by difference magnitude
    _, top_indices = torch.abs(diff).topk(NUM_FEATURES)
    
    # Create sparse steering vector
    steering_vector = torch.zeros_like(diff)
    steering_vector[top_indices] = diff[top_indices]
    
    return steering_vector


def run_spillover_experiment(
    model,
    steering_vector_features,
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
    
    # CRITICAL: Convert steering vector from SAE feature space (16384) to hidden space (2304)
    # The SAE decoder maps features -> hidden states
    sae = model.load_sae(layer)
    
    # Project through decoder: (16384,) -> (2304,)
    # The decoder is W_dec with shape (d_sae, d_model) = (16384, 2304)
    steering_vector_hidden = sae.decode(steering_vector_features.unsqueeze(0)).squeeze(0)
    print(f"  Steering vector converted: {steering_vector_features.shape} -> {steering_vector_hidden.shape}")
    
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
    """Run spillover experiment."""
    print("=" * 60)
    print("EXPERIMENT 4: Language Steering Spillover Analysis")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = GemmaWithSAE()
    model.load_model()
    
    # Load data
    print("Loading FLORES-200...")
    flores = load_flores(max_samples=200)
    texts_hi = flores.get("hi", [])
    texts_en = flores.get("en", [])
    
    if not texts_hi or not texts_en:
        print("ERROR: Could not load Hindi or English data!")
        return
    
    # Use evaluation prompts
    prompts = EVAL_PROMPTS[:20]  # Use subset for speed
    print(f"Using {len(prompts)} evaluation prompts")
    
    all_results = {}
    
    # Test at different layers
    test_layers = [13, 20]  # Mid and late layers
    
    for layer in test_layers:
        print(f"\n{'='*20} Layer {layer} {'='*20}")
        
        # Compute steering vector
        print("Computing steering vector (EN → HI)...")
        steering_vector = compute_steering_vector(model, texts_hi, texts_en, layer)
        
        # Run spillover experiment
        print("Running spillover experiment...")
        layer_results = run_spillover_experiment(
            model, steering_vector, layer, prompts,
            strengths=[0.5, 1.0, 2.0, 4.0],
            target_lang="hi"
        )
        
        all_results[layer] = layer_results
        
        # Print spillover matrix
        print(f"\n  SPILLOVER MATRIX (Layer {layer}):")
        print("  " + "-" * 50)
        print(f"  {'Strength':<10} {'Hindi':<8} {'Urdu':<8} {'Bengali':<8} {'English':<8} {'German':<8}")
        print("  " + "-" * 50)
        
        for strength in sorted(layer_results.keys()):
            result = layer_results[strength]
            dist = result.language_distribution
            print(f"  {strength:<10} {dist.get('hi', 0):<8.1f} {dist.get('ur', 0):<8.1f} "
                  f"{dist.get('bn', 0):<8.1f} {dist.get('en', 0):<8.1f} {dist.get('de', 0):<8.1f}")
    
    # Hypothesis tests on best layer
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTS")
    print("=" * 60)
    
    # Use layer 20 for tests (typically best for steering)
    test_layer = 20 if 20 in all_results else list(all_results.keys())[0]
    hypothesis_results = test_hypotheses(all_results[test_layer])
    
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
        "layers": {
            str(layer): {
                str(strength): {
                    "language_distribution": result.language_distribution,
                    "n_samples": result.n_samples,
                }
                for strength, result in layer_results.items()
            }
            for layer, layer_results in all_results.items()
        },
        "hypothesis_tests": hypothesis_results,
        "family_analysis": analyze_spillover_by_family(all_results[test_layer]),
    }
    
    with open(output_dir / "exp4_spillover.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'exp4_spillover.json'}")
    
    return all_results


if __name__ == "__main__":
    main()
