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

import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import re

from config import (
    TARGET_LAYERS,
    LAYER_RANGES,
    STEERING_STRENGTHS,
    NUM_FEATURES,
    EVAL_PROMPTS,
    SCRIPT_RANGES,
    N_SAMPLES_EVAL,
    MIN_PROMPTS_STEERING,
)
from data import load_flores, load_research_data
from model import GemmaWithSAE
from evaluation_comprehensive import (
    detect_language,
    get_language_with_confidence,
    LID_CONFIDENCE_THRESHOLD,
)


@dataclass
class SpilloverResult:
    """Results for a single steering configuration."""
    target_lang: str
    layer: int
    strength: float
    num_features: int
    language_distribution: Dict[str, float]  # Lang -> % of outputs
    n_samples: int
    # LID confidence metrics
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    n_reliable: int = 0  # Count of samples with confidence >= threshold
    n_unreliable: int = 0  # Count of low-confidence detections


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
    """Language detection wrapper.

    We delegate to the shared evaluator's detect_language() so Exp4 uses
    the same script dominance and Latin-language heuristics as Exp9/12.
    """
    lang, _conf = detect_language(text)
    return lang


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
        target_acts.append(acts.mean(dim=0).detach())
    target_mean = torch.stack(target_acts).mean(dim=0)
    del target_acts

    # Get mean activations for source language
    source_acts = []
    for text in texts_source[:100]:
        acts = model.get_sae_activations(text, layer)
        source_acts.append(acts.mean(dim=0).detach())
    source_mean = torch.stack(source_acts).mean(dim=0)
    del source_acts

    # Difference: positive values correspond to features more active for target.
    diff = target_mean - source_mean
    diff_clamped = torch.clamp(diff, min=0.0)

    # Select top-k features by positive difference. If, for some reason, no
    # feature is strictly more active for the target language, we fall back
    # to a dense EN→HI steering vector in hidden space. Returning a zero
    # vector here would silently disable steering and produce 100% English
    # outputs, which is exactly the suspicious pattern we want to avoid.
    if (diff_clamped > 0).sum() == 0:
        from steering_utils import construct_dense_steering_vector

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

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        steering_vector_hidden: Steering vector in hidden space
        layer: Layer to steer
        prompts: Prompts to test
        strengths: Steering strengths
        target_lang: Target language code

    Returns:
        Dict mapping strength -> SpilloverResult with confidence-aware metrics
    """
    results = {}

    # Test each strength (including 0.0 for baseline)
    test_strengths = [0.0] + list(strengths)

    for strength in test_strengths:
        print(f"\n  Testing strength {strength}...")
        lang_counts = defaultdict(int)
        confidences = []
        n_reliable = 0
        n_unreliable = 0

        for prompt in prompts:
            if strength == 0.0:
                # Baseline: no steering
                output = model.generate(prompt, max_new_tokens=64)
            else:
                # With steering (using hidden-space vector)
                output = model.generate_with_steering(
                    prompt, layer, steering_vector_hidden, strength
                )

            # Detect output language WITH confidence
            detected_lang, conf, is_reliable = get_language_with_confidence(output)
            lang_counts[detected_lang] += 1
            confidences.append(conf)

            if is_reliable:
                n_reliable += 1
            else:
                n_unreliable += 1

        # Convert to percentages
        total = len(prompts)
        lang_distribution = {
            lang: count / total * 100
            for lang, count in lang_counts.items()
        }

        # Compute confidence statistics
        conf_mean = float(np.mean(confidences)) if confidences else 0.0
        conf_std = float(np.std(confidences)) if confidences else 0.0

        results[strength] = SpilloverResult(
            target_lang=target_lang,
            layer=layer,
            strength=strength,
            num_features=NUM_FEATURES,
            language_distribution=lang_distribution,
            n_samples=total,
            confidence_mean=conf_mean,
            confidence_std=conf_std,
            n_reliable=n_reliable,
            n_unreliable=n_unreliable,
        )

        # Print summary with confidence info
        reliable_pct = 100 * n_reliable / total if total > 0 else 0
        print(f"    Results: {dict(lang_distribution)}")
        print(f"    LID confidence: {conf_mean:.2f} ± {conf_std:.2f}, reliable: {reliable_pct:.0f}%")

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
    suffix = "_9b" if "9b" in str(getattr(model, "model_id", "")).lower() else ""
    
    # Load data using the unified research loader so that training
    # sentences (for steering vectors) and evaluation prompts are
    # properly separated.
    print("Loading research data for spillover experiment...")
    data_split = load_research_data()
    train_data = data_split.train
    prompts = data_split.steering_prompts

    lid_backend = os.environ.get("LID_BACKEND", "regex").strip().lower()
    if lid_backend != "fasttext":
        print(
            "[exp4] Note: LID_BACKEND is not 'fasttext'. "
            "Latin/Arabic-script outputs may be misclassified by lightweight heuristics. "
            "For shared-script controls, prefer LID_BACKEND=fasttext."
        )

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
    
    # Test at representative middle and late layers for both EN→HI and EN→DE.
    # We pull these from LAYER_RANGES so 9B runs probe comparable relative depth.
    try:
        mid_layer = LAYER_RANGES.get("middle", [TARGET_LAYERS[len(TARGET_LAYERS)//2]])[1]
    except Exception:
        mid_layer = TARGET_LAYERS[len(TARGET_LAYERS)//2]
    try:
        late_layer = LAYER_RANGES.get("late", [TARGET_LAYERS[-1]])[1]
    except Exception:
        late_layer = TARGET_LAYERS[-1]
    test_layers = [mid_layer, late_layer]
    
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
    
    # Convert to JSON-serializable format with confidence metrics
    json_results = {
        "lid_backend": lid_backend,
        "lid_confidence_threshold": LID_CONFIDENCE_THRESHOLD,
        "en_to_hi_layers": {
            str(layer): {
                str(strength): {
                    "language_distribution": result.language_distribution,
                    "n_samples": result.n_samples,
                    "lid_confidence_mean": result.confidence_mean,
                    "lid_confidence_std": result.confidence_std,
                    "n_reliable_detections": result.n_reliable,
                    "n_unreliable_detections": result.n_unreliable,
                    "reliability_rate": result.n_reliable / result.n_samples if result.n_samples > 0 else 0,
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
                    "lid_confidence_mean": result.confidence_mean,
                    "lid_confidence_std": result.confidence_std,
                    "n_reliable_detections": result.n_reliable,
                    "n_unreliable_detections": result.n_unreliable,
                    "reliability_rate": result.n_reliable / result.n_samples if result.n_samples > 0 else 0,
                }
                for strength, result in layer_results.items()
            }
            for layer, layer_results in all_results["en_to_de"].items()
        },
        "hypothesis_tests": hypothesis_results,
        "family_analysis_en_to_hi": analyze_spillover_by_family(all_results["en_to_hi"][test_layer]),
        "family_analysis_en_to_de": analyze_spillover_by_family(all_results["en_to_de"][test_layer]),
    }
    
    out_path = output_dir / f"exp4_spillover{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {out_path}")
    
    return all_results


if __name__ == "__main__":
    main()
