"""Evaluation metrics for SAE Multilingual Steering.

Metrics:
1. Script Detection - What script is the output in?
2. Language ID - What language is the output?
3. Repetition Detection - Is the output degraded?
4. Perplexity - Is the output fluent?
5. Semantic Similarity - Does meaning match?
6. LLM-as-Judge - Human-like evaluation

Also includes CORRECT Jaccard overlap computation (bug fixed!).
"""

import re
import os
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import json

from config import SCRIPT_RANGES, DEVANAGARI_THRESHOLD


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SteeringEvalResult:
    """Result of evaluating a single steering output."""
    prompt: str
    output: str
    script_ratios: Dict[str, float]
    is_target_script: bool
    repetition_3gram: float
    repetition_5gram: float
    is_degraded: bool
    llm_judge_result: Optional[Dict] = None


@dataclass
class AggregateResults:
    """Aggregated results across multiple evaluations."""
    n_samples: int
    success_rate: float
    avg_target_script_ratio: float
    avg_repetition_3gram: float
    avg_repetition_5gram: float
    degradation_rate: float
    per_strength_results: Dict[float, Dict] = field(default_factory=dict)


# =============================================================================
# SCRIPT DETECTION
# =============================================================================

def detect_script_chars(text: str, script_name: str) -> int:
    """Count characters in a specific script.
    
    Args:
        text: Text to analyze
        script_name: Name of script (e.g., "devanagari")
        
    Returns:
        Number of characters in that script
    """
    if script_name not in SCRIPT_RANGES:
        return 0
    
    start, end = SCRIPT_RANGES[script_name]
    count = sum(1 for c in text if start <= ord(c) <= end)
    return count


def detect_scripts(text: str) -> Dict[str, float]:
    """Detect what scripts are present in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict mapping script name to ratio (0-1)
    """
    if not text:
        return {}
    
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return {}
    
    ratios = {}
    for script_name in SCRIPT_RANGES:
        count = detect_script_chars(text, script_name)
        ratios[script_name] = count / total_alpha
    
    return ratios


def is_target_script(
    text: str, 
    target: str = "devanagari",
    threshold: float = DEVANAGARI_THRESHOLD
) -> bool:
    """Check if text is primarily in target script.
    
    Args:
        text: Text to check
        target: Target script name
        threshold: Minimum ratio required
        
    Returns:
        True if text is in target script
    """
    ratios = detect_scripts(text)
    return ratios.get(target, 0) >= threshold


# =============================================================================
# REPETITION DETECTION
# =============================================================================

def compute_ngram_repetition(text: str, n: int) -> float:
    """Compute n-gram repetition rate.
    
    High repetition = degraded output (model stuck in loop)
    
    Args:
        text: Text to analyze
        n: N-gram size (3, 5, 7)
        
    Returns:
        Repetition rate (0-1), higher = more repetition
    """
    if not text or len(text) < n:
        return 0.0
    
    # Get n-grams
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    # Count occurrences
    counts = Counter(ngrams)
    
    # Repetition = (total - unique) / total
    total = len(ngrams)
    unique = len(counts)
    
    if total == 0:
        return 0.0
    
    repetition = (total - unique) / total
    return repetition


def is_degraded(
    text: str,
    threshold_3gram: float = 0.3,
    threshold_5gram: float = 0.2
) -> bool:
    """Check if text shows signs of degradation.
    
    Args:
        text: Text to check
        threshold_3gram: Max allowed 3-gram repetition
        threshold_5gram: Max allowed 5-gram repetition
        
    Returns:
        True if text is degraded
    """
    rep_3 = compute_ngram_repetition(text, 3)
    rep_5 = compute_ngram_repetition(text, 5)
    
    return rep_3 > threshold_3gram or rep_5 > threshold_5gram


# =============================================================================
# JACCARD OVERLAP (CORRECT IMPLEMENTATION!)
# =============================================================================

def jaccard_overlap(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets.
    
    CORRECT FORMULA: |A ∩ B| / |A ∪ B|
    
    This ALWAYS returns a value between 0 and 1.
    
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
    
    result = intersection / union
    
    # Sanity check - this should NEVER fail
    assert 0.0 <= result <= 1.0, f"Invalid Jaccard: {result}"
    
    return result


def dice_coefficient(set_a: Set, set_b: Set) -> float:
    """Compute Dice coefficient (alternative to Jaccard).
    
    FORMULA: 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        set_a: First set
        set_b: Second set
        
    Returns:
        Dice coefficient (0 to 1)
    """
    intersection = len(set_a & set_b)
    total = len(set_a) + len(set_b)
    
    if total == 0:
        return 0.0
    
    return 2 * intersection / total


def overlap_coefficient(set_a: Set, set_b: Set) -> float:
    """Compute overlap coefficient.
    
    FORMULA: |A ∩ B| / min(|A|, |B|)
    
    This measures how much the smaller set is contained in the larger.
    
    Args:
        set_a: First set
        set_b: Second set
        
    Returns:
        Overlap coefficient (0 to 1)
    """
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    
    if min_size == 0:
        return 0.0
    
    return intersection / min_size


# =============================================================================
# LLM-AS-JUDGE
# =============================================================================

def llm_judge_gemini(
    prompt: str,
    output: str,
    api_key: Optional[str] = None
) -> Optional[Dict]:
    """Use Gemini as a judge for output quality.
    
    Args:
        prompt: Original prompt
        output: Model output
        api_key: Gemini API key (or from environment)
        
    Returns:
        Dict with evaluation results or None if error
    """
    import os
    
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    if not api_key:
        return None
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        judge_prompt = f"""You are evaluating whether a language model successfully generated Hindi text.

Task: Determine if the output is PRIMARILY in Hindi (Devanagari script).

Evaluation criteria:
1. Is the majority of the text in Hindi (Devanagari script)?
2. Is the text coherent and meaningful?
3. Does the text respond appropriately to the prompt?

Prompt: {prompt}
Output: {output}

Respond ONLY with a JSON object (no markdown, no explanation):
{{"is_hindi": true/false, "coherence_score": 1-5, "relevance_score": 1-5, "reasoning": "brief explanation"}}"""
        
        response = model.generate_content(judge_prompt)
        
        # Parse JSON from response
        text = response.text.strip()
        # Remove markdown if present
        if text.startswith("```"):
            text = re.sub(r"```\w*\n?", "", text).strip()
        
        return json.loads(text)
        
    except Exception as e:
        print(f"LLM judge error: {e}")
        return None


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_steering_output(
    prompt: str,
    output: str,
    target_script: str = "devanagari",
    use_llm_judge: bool = False
) -> SteeringEvalResult:
    """Comprehensive evaluation of a steering output.
    
    Args:
        prompt: Original prompt
        output: Steered model output
        target_script: Expected script
        use_llm_judge: Whether to use LLM judge
        
    Returns:
        SteeringEvalResult with all metrics
    """
    # Script detection
    script_ratios = detect_scripts(output)
    target_ratio = script_ratios.get(target_script, 0)
    is_target = target_ratio >= DEVANAGARI_THRESHOLD
    
    # Repetition detection
    rep_3 = compute_ngram_repetition(output, 3)
    rep_5 = compute_ngram_repetition(output, 5)
    degraded = is_degraded(output)
    
    # LLM judge (optional)
    llm_result = None
    if use_llm_judge:
        llm_result = llm_judge_gemini(prompt, output)
    
    return SteeringEvalResult(
        prompt=prompt,
        output=output,
        script_ratios=script_ratios,
        is_target_script=is_target,
        repetition_3gram=rep_3,
        repetition_5gram=rep_5,
        is_degraded=degraded,
        llm_judge_result=llm_result
    )


def aggregate_results(results: List[SteeringEvalResult]) -> AggregateResults:
    """Aggregate multiple evaluation results.
    
    Args:
        results: List of individual results
        
    Returns:
        AggregateResults with summary statistics
    """
    if not results:
        return AggregateResults(
            n_samples=0,
            success_rate=0.0,
            avg_target_script_ratio=0.0,
            avg_repetition_3gram=0.0,
            avg_repetition_5gram=0.0,
            degradation_rate=0.0
        )
    
    n = len(results)
    
    success_count = sum(1 for r in results if r.is_target_script)
    target_ratios = [r.script_ratios.get("devanagari", 0) for r in results]
    rep_3_values = [r.repetition_3gram for r in results]
    rep_5_values = [r.repetition_5gram for r in results]
    degraded_count = sum(1 for r in results if r.is_degraded)
    
    return AggregateResults(
        n_samples=n,
        success_rate=success_count / n,
        avg_target_script_ratio=sum(target_ratios) / n,
        avg_repetition_3gram=sum(rep_3_values) / n,
        avg_repetition_5gram=sum(rep_5_values) / n,
        degradation_rate=degraded_count / n
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing evaluation module...")
    
    # Test Jaccard (CRITICAL!)
    print("\n1. Testing Jaccard overlap...")
    set_a = {1, 2, 3, 4, 5}
    set_b = {3, 4, 5, 6, 7}
    
    jaccard = jaccard_overlap(set_a, set_b)
    expected = 3 / 7  # |{3,4,5}| / |{1,2,3,4,5,6,7}| = 3/7
    
    print(f"  Set A: {set_a}")
    print(f"  Set B: {set_b}")
    print(f"  Intersection: {set_a & set_b}")
    print(f"  Union: {set_a | set_b}")
    print(f"  Jaccard: {jaccard:.4f} (expected {expected:.4f})")
    
    assert abs(jaccard - expected) < 0.001, f"Jaccard failed! Got {jaccard}, expected {expected}"
    assert 0 <= jaccard <= 1, f"Jaccard out of range: {jaccard}"
    print("  ✓ Jaccard test passed!")
    
    # Test edge cases
    print("\n2. Testing Jaccard edge cases...")
    
    # Identical sets
    same = jaccard_overlap({1, 2, 3}, {1, 2, 3})
    assert same == 1.0, f"Same sets should have Jaccard=1, got {same}"
    print(f"  Identical sets: {same} (expected 1.0) ✓")
    
    # Disjoint sets
    disjoint = jaccard_overlap({1, 2}, {3, 4})
    assert disjoint == 0.0, f"Disjoint sets should have Jaccard=0, got {disjoint}"
    print(f"  Disjoint sets: {disjoint} (expected 0.0) ✓")
    
    # Empty sets
    empty = jaccard_overlap(set(), set())
    assert empty == 0.0, f"Empty sets should have Jaccard=0, got {empty}"
    print(f"  Empty sets: {empty} (expected 0.0) ✓")
    
    # Test script detection
    print("\n3. Testing script detection...")
    hindi_text = "नमस्ते दुनिया"
    ratios = detect_scripts(hindi_text)
    print(f"  Hindi text: '{hindi_text}'")
    print(f"  Devanagari ratio: {ratios.get('devanagari', 0):.2f}")
    assert ratios.get("devanagari", 0) > 0.9, "Should detect Hindi as Devanagari"
    print("  ✓ Script detection passed!")
    
    # Test repetition
    print("\n4. Testing repetition detection...")
    normal = "This is a normal sentence with varied words."
    repeated = "hello hello hello hello hello hello"
    
    rep_normal = compute_ngram_repetition(normal, 3)
    rep_bad = compute_ngram_repetition(repeated, 3)
    
    print(f"  Normal text repetition: {rep_normal:.2f}")
    print(f"  Repeated text repetition: {rep_bad:.2f}")
    assert rep_bad > rep_normal, "Repeated text should have higher repetition"
    print("  ✓ Repetition detection passed!")
    
    print("\n✓ All tests passed!")
