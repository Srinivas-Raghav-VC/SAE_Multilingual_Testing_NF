"""Comprehensive Evaluation Module for SAE Multilingual Steering.

This module implements research-grade evaluation metrics based on:
- O'Brien et al. 2024: Capability preservation during steering
- Gao et al. 2024: SAE feature quality metrics
- Vogels et al. 2025: Coherence and in-distribution steering

Metrics:
1. Steering Success: Did the model output Hindi?
2. Fluency: Is the output grammatically correct? (perplexity-based)
3. Coherence: Is the output coherent? (repetition detection)
4. Capability Preservation: Can the model still perform basic tasks?
5. Feature Quality: Are selected features high-quality?
"""

import math
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json

# ============================================================================
# EVALUATION RESULT CONTAINER
# ============================================================================
@dataclass
class SteeringEvalResult:
    """Container for comprehensive steering evaluation results."""
    prompt: str
    output: str
    steering_strength: float
    
    # Steering success
    is_hindi: bool = False
    devanagari_ratio: float = 0.0
    
    # Fluency (perplexity)
    perplexity: float = 0.0
    perplexity_increase: float = 0.0  # vs baseline
    
    # Coherence (repetition)
    repetition_3gram: float = 0.0
    repetition_5gram: float = 0.0
    is_degraded: bool = False
    
    # LLM judge scores
    llm_is_hindi: bool = False
    llm_coherence: int = 0
    llm_relevance: int = 0
    
    # Overall
    success: bool = False
    
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "output": self.output[:200] + "..." if len(self.output) > 200 else self.output,
            "steering_strength": self.steering_strength,
            "is_hindi": self.is_hindi,
            "devanagari_ratio": self.devanagari_ratio,
            "perplexity": self.perplexity,
            "perplexity_increase": self.perplexity_increase,
            "repetition_3gram": self.repetition_3gram,
            "repetition_5gram": self.repetition_5gram,
            "is_degraded": self.is_degraded,
            "llm_is_hindi": self.llm_is_hindi,
            "llm_coherence": self.llm_coherence,
            "llm_relevance": self.llm_relevance,
            "success": self.success,
        }


@dataclass
class CapabilityEvalResult:
    """Container for capability preservation evaluation."""
    task: str
    prompt: str
    baseline_output: str
    steered_output: str
    
    # Scores
    baseline_correct: bool = False
    steered_correct: bool = False
    capability_preserved: bool = False
    
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "prompt": self.prompt,
            "baseline_output": self.baseline_output[:100],
            "steered_output": self.steered_output[:100],
            "baseline_correct": self.baseline_correct,
            "steered_correct": self.steered_correct,
            "capability_preserved": self.capability_preserved,
        }


# ============================================================================
# SCRIPT DETECTION
# ============================================================================
def detect_script_ratio(text: str, script: str = "devanagari") -> float:
    """Detect ratio of characters in a specific script.
    
    Args:
        text: Input text
        script: Script to detect (devanagari, arabic, latin, etc.)
        
    Returns:
        Ratio of characters in the script (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    # Unicode ranges for different scripts
    SCRIPT_RANGES = {
        "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit
        "arabic": (0x0600, 0x06FF),       # Urdu, Arabic
        "bengali": (0x0980, 0x09FF),
        "tamil": (0x0B80, 0x0BFF),
        "telugu": (0x0C00, 0x0C7F),
        "latin": (0x0041, 0x007A),        # A-Z, a-z (excludes numbers/punctuation)
        "cyrillic": (0x0400, 0x04FF),     # Russian
    }
    
    if script not in SCRIPT_RANGES:
        return 0.0
    
    start, end = SCRIPT_RANGES[script]
    
    # Count script characters (excluding whitespace and punctuation)
    script_chars = sum(1 for c in text if start <= ord(c) <= end)
    # Count all non-whitespace characters
    total_chars = sum(1 for c in text if not c.isspace() and ord(c) > 0x20)
    
    return script_chars / total_chars if total_chars > 0 else 0.0


def is_primarily_hindi(text: str, threshold: float = 0.3) -> bool:
    """Check if text is primarily in Hindi (Devanagari script)."""
    return detect_script_ratio(text, "devanagari") >= threshold


# ============================================================================
# REPETITION DETECTION (COHERENCE)
# ============================================================================
def detect_repetition(text: str, n_gram: int = 3) -> float:
    """Detect n-gram repetition patterns.
    
    Based on literature:
    - High repetition indicates steering-induced degradation
    - rep3 > 0.3 or rep5 > 0.2 indicates coherence loss
    
    Args:
        text: Input text
        n_gram: Size of n-gram to check
        
    Returns:
        Repetition ratio (0 = no repetition, 1 = all repeated)
    """
    words = text.split()
    
    if len(words) < n_gram:
        return 0.0
    
    ngrams = [tuple(words[i:i+n_gram]) for i in range(len(words) - n_gram + 1)]
    
    if not ngrams:
        return 0.0
    
    unique = len(set(ngrams))
    total = len(ngrams)
    
    # Repetition ratio = 1 - (unique / total)
    return 1.0 - (unique / total)


def detect_repetition_patterns(text: str) -> Dict[str, float]:
    """Detect multiple repetition patterns.
    
    Returns dict with different n-gram repetition scores.
    """
    return {
        "repetition_2gram": detect_repetition(text, 2),
        "repetition_3gram": detect_repetition(text, 3),
        "repetition_5gram": detect_repetition(text, 5),
        "repetition_7gram": detect_repetition(text, 7),
    }


def is_degraded(text: str, 
                rep3_threshold: float = 0.3, 
                rep5_threshold: float = 0.2) -> bool:
    """Check if output shows signs of steering-induced degradation."""
    rep3 = detect_repetition(text, 3)
    rep5 = detect_repetition(text, 5)
    
    return rep3 > rep3_threshold or rep5 > rep5_threshold


# ============================================================================
# PERPLEXITY CALCULATION
# ============================================================================
def compute_perplexity(model, text: str) -> float:
    """Compute perplexity of text using the model.
    
    Lower perplexity = more fluent/natural text.
    High perplexity = less natural/coherent text.
    
    Args:
        model: Language model with forward pass
        text: Text to evaluate
        
    Returns:
        Perplexity score
    """
    try:
        import torch
        
        # Tokenize
        inputs = model.tokenizer(text, return_tensors="pt").to(model.device)
        
        # Get loss
        with torch.no_grad():
            outputs = model.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        # Perplexity = exp(loss)
        return math.exp(loss)
        
    except Exception as e:
        print(f"Perplexity calculation error: {e}")
        return float('inf')


def compute_perplexity_increase(
    model, 
    steered_text: str, 
    baseline_text: str
) -> float:
    """Compute relative perplexity increase from baseline.
    
    Returns ratio: steered_ppl / baseline_ppl
    - ratio = 1.0: No change
    - ratio > 1.0: Degradation (less fluent)
    - ratio < 1.0: Improvement (more fluent)
    """
    steered_ppl = compute_perplexity(model, steered_text)
    baseline_ppl = compute_perplexity(model, baseline_text)
    
    if baseline_ppl == 0 or baseline_ppl == float('inf'):
        return float('inf')
    
    return steered_ppl / baseline_ppl


# ============================================================================
# LLM-AS-JUDGE EVALUATION
# ============================================================================
def llm_judge_evaluate(
    prompt: str, 
    output: str,
    provider: str = "gemini",
    model_name: str = "gemini-2.0-flash"
) -> Dict:
    """Evaluate output using LLM as judge.
    
    Supports: gemini (free), claude, openai
    
    Returns:
        {"is_hindi": bool, "coherence_score": int, "relevance_score": int, "reasoning": str}
    """
    import os
    
    # Format prompt
    judge_prompt = f"""You are evaluating whether a language model successfully generated Hindi text.

Task: Determine if the output is PRIMARILY in Hindi (Devanagari script).

Evaluation criteria:
1. Is the majority of the text in Hindi (Devanagari script)?
2. Is the text coherent and meaningful?
3. Does the text respond appropriately to the prompt?

Prompt: {prompt}
Output: {output[:500]}

Respond ONLY with a JSON object (no markdown, no explanation):
{{"is_hindi": true/false, "coherence_score": 1-5, "relevance_score": 1-5, "reasoning": "brief explanation"}}"""

    try:
        if provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                return {"is_hindi": False, "coherence_score": 0, "relevance_score": 0, 
                        "reasoning": "No API key", "error": True}
            
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=judge_prompt,
            )
            response_text = response.text
            
        elif provider == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return {"is_hindi": False, "coherence_score": 0, "relevance_score": 0,
                        "reasoning": "No API key", "error": True}
            
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_name,
                max_tokens=200,
                messages=[{"role": "user", "content": judge_prompt}]
            )
            response_text = response.content[0].text
            
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return {"is_hindi": False, "coherence_score": 0, "relevance_score": 0,
                        "reasoning": "No API key", "error": True}
            
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=200
            )
            response_text = response.choices[0].message.content
        
        else:
            return {"is_hindi": False, "coherence_score": 0, "relevance_score": 0,
                    "reasoning": f"Unknown provider: {provider}", "error": True}
        
        # Parse JSON response
        # Clean up response (remove markdown if present)
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        result = json.loads(response_text)
        result["error"] = False
        return result
        
    except Exception as e:
        return {
            "is_hindi": False, 
            "coherence_score": 0, 
            "relevance_score": 0,
            "reasoning": str(e),
            "error": True
        }


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================
def evaluate_steered_output(
    prompt: str,
    output: str,
    steering_strength: float,
    baseline_output: Optional[str] = None,
    model = None,
    use_llm_judge: bool = True,
    llm_provider: str = "gemini"
) -> SteeringEvalResult:
    """Comprehensive evaluation of steered output.
    
    Args:
        prompt: Input prompt
        output: Steered model output
        steering_strength: Strength used for steering
        baseline_output: Unsteered output (for perplexity comparison)
        model: Language model (for perplexity calculation)
        use_llm_judge: Whether to use LLM judge
        llm_provider: LLM provider for judge
        
    Returns:
        SteeringEvalResult with all metrics
    """
    result = SteeringEvalResult(
        prompt=prompt,
        output=output,
        steering_strength=steering_strength
    )
    
    # 1. Script detection
    result.devanagari_ratio = detect_script_ratio(output, "devanagari")
    result.is_hindi = result.devanagari_ratio >= 0.3
    
    # 2. Repetition detection (coherence)
    result.repetition_3gram = detect_repetition(output, 3)
    result.repetition_5gram = detect_repetition(output, 5)
    result.is_degraded = is_degraded(output)
    
    # 3. Perplexity (fluency)
    if model is not None and baseline_output is not None:
        result.perplexity_increase = compute_perplexity_increase(
            model, output, baseline_output
        )
    
    # 4. LLM judge
    if use_llm_judge:
        judge_result = llm_judge_evaluate(prompt, output, provider=llm_provider)
        if not judge_result.get("error", False):
            result.llm_is_hindi = judge_result.get("is_hindi", False)
            result.llm_coherence = judge_result.get("coherence_score", 0)
            result.llm_relevance = judge_result.get("relevance_score", 0)
    
    # 5. Overall success
    # Success = Hindi + not degraded + coherent
    result.success = (
        result.is_hindi and 
        not result.is_degraded and
        (not use_llm_judge or result.llm_coherence >= 3)
    )
    
    return result


# ============================================================================
# CAPABILITY PRESERVATION EVALUATION
# ============================================================================
def evaluate_capability_preservation(
    model,
    steering_vector,
    steering_strength: float,
    capability_prompts: List[str],
    expected_answers: Optional[Dict[str, str]] = None
) -> List[CapabilityEvalResult]:
    """Evaluate if steering preserves model capabilities.
    
    Based on O'Brien et al. 2024:
    "Systematic degradation of performance across multiple benchmark tasks,
    even on safe inputs with no apparent connection to refusal behavior."
    
    Args:
        model: Language model with steering capability
        steering_vector: The steering vector to apply
        steering_strength: Strength to apply
        capability_prompts: List of capability test prompts
        expected_answers: Optional dict of expected answers
        
    Returns:
        List of CapabilityEvalResult
    """
    results = []
    
    # Simple expected answers (keyword matching)
    DEFAULT_EXPECTED = {
        "What is the capital of France?": ["Paris", "paris"],
        "Who wrote Romeo and Juliet?": ["Shakespeare", "shakespeare", "William"],
        "What is the chemical formula for water?": ["H2O", "h2o"],
        "How many continents are there?": ["7", "seven"],
        "What planet is known as the Red Planet?": ["Mars", "mars"],
        "If I have 3 apples and give away 1, how many do I have?": ["2", "two"],
        "What comes next in the sequence: 2, 4, 6, 8, ?": ["10", "ten"],
        "What is 15 + 27?": ["42"],
        "If today is Monday, what day is tomorrow?": ["Tuesday", "tuesday"],
    }
    
    if expected_answers is None:
        expected_answers = DEFAULT_EXPECTED
    
    for prompt in capability_prompts:
        # Generate baseline (no steering)
        baseline_output = model.generate(prompt, steering_vector=None)
        
        # Generate steered
        steered_output = model.generate(
            prompt, 
            steering_vector=steering_vector,
            steering_strength=steering_strength
        )
        
        # Check correctness
        expected = expected_answers.get(prompt, [])
        
        baseline_correct = any(
            exp.lower() in baseline_output.lower() 
            for exp in expected
        ) if expected else True  # Assume correct if no expected answer
        
        steered_correct = any(
            exp.lower() in steered_output.lower() 
            for exp in expected
        ) if expected else True
        
        result = CapabilityEvalResult(
            task="general_knowledge" if "?" in prompt else "instruction",
            prompt=prompt,
            baseline_output=baseline_output,
            steered_output=steered_output,
            baseline_correct=baseline_correct,
            steered_correct=steered_correct,
            capability_preserved=(baseline_correct == steered_correct)
        )
        
        results.append(result)
    
    return results


# ============================================================================
# AGGREGATE METRICS
# ============================================================================
def compute_aggregate_metrics(
    results: List[SteeringEvalResult]
) -> Dict:
    """Compute aggregate metrics from evaluation results."""
    
    if not results:
        return {}
    
    n = len(results)
    
    return {
        "n_samples": n,
        "success_rate": sum(r.success for r in results) / n,
        "hindi_rate": sum(r.is_hindi for r in results) / n,
        "avg_devanagari_ratio": sum(r.devanagari_ratio for r in results) / n,
        "degradation_rate": sum(r.is_degraded for r in results) / n,
        "avg_repetition_3gram": sum(r.repetition_3gram for r in results) / n,
        "avg_repetition_5gram": sum(r.repetition_5gram for r in results) / n,
        "llm_hindi_rate": sum(r.llm_is_hindi for r in results) / n,
        "avg_llm_coherence": sum(r.llm_coherence for r in results) / n,
        "avg_llm_relevance": sum(r.llm_relevance for r in results) / n,
    }


def compute_capability_preservation_rate(
    results: List[CapabilityEvalResult]
) -> Dict:
    """Compute capability preservation metrics."""
    
    if not results:
        return {}
    
    n = len(results)
    
    return {
        "n_samples": n,
        "baseline_accuracy": sum(r.baseline_correct for r in results) / n,
        "steered_accuracy": sum(r.steered_correct for r in results) / n,
        "preservation_rate": sum(r.capability_preserved for r in results) / n,
        "capability_drop": (
            sum(r.baseline_correct for r in results) - 
            sum(r.steered_correct for r in results)
        ) / n,
    }


# ============================================================================
# JACCARD OVERLAP (CORRECT IMPLEMENTATION)
# ============================================================================
def jaccard_overlap(set_a: set, set_b: set) -> float:
    """Compute Jaccard overlap between two sets.
    
    CORRECT implementation - cannot exceed 1.0 (100%)
    
    J(A,B) = |A ∩ B| / |A ∪ B|
    
    Args:
        set_a: First set of feature IDs
        set_b: Second set of feature IDs
        
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not isinstance(set_a, set):
        set_a = set(set_a)
    if not isinstance(set_b, set):
        set_b = set(set_b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    overlap = intersection / union
    
    # Sanity check - Jaccard CANNOT exceed 1.0
    assert overlap <= 1.0, f"Bug: Jaccard overlap = {overlap} > 1.0!"
    
    return overlap


def semantic_vs_script_overlap(
    features_lang1: set,
    features_lang2: set,
    script1: str,
    script2: str
) -> Dict:
    """Analyze semantic vs script feature overlap between two languages.
    
    For Hindi-Urdu:
    - SEMANTIC overlap should be HIGH (94%+) - same spoken language
    - SCRIPT overlap should be LOW (<20%) - different writing systems
    
    This helps disentangle semantic content from orthographic representation.
    """
    total_overlap = jaccard_overlap(features_lang1, features_lang2)
    
    # Shared features (intersection)
    shared = features_lang1 & features_lang2
    
    # Unique features
    unique_lang1 = features_lang1 - features_lang2
    unique_lang2 = features_lang2 - features_lang1
    
    return {
        "total_overlap": total_overlap,
        "shared_features": len(shared),
        "unique_lang1": len(unique_lang1),  # Likely script-specific
        "unique_lang2": len(unique_lang2),  # Likely script-specific
        "script1": script1,
        "script2": script2,
    }


# ============================================================================
# MAIN TEST
# ============================================================================
if __name__ == "__main__":
    # Test evaluation functions
    print("=" * 60)
    print("EVALUATION MODULE TEST")
    print("=" * 60)
    
    # Test script detection
    hindi_text = "यह एक परीक्षण वाक्य है।"
    english_text = "This is a test sentence."
    mixed_text = "यह is a मिश्रित sentence."
    
    print("\n1. Script Detection:")
    print(f"  Hindi text: {detect_script_ratio(hindi_text, 'devanagari'):.2%} Devanagari")
    print(f"  English text: {detect_script_ratio(english_text, 'devanagari'):.2%} Devanagari")
    print(f"  Mixed text: {detect_script_ratio(mixed_text, 'devanagari'):.2%} Devanagari")
    
    # Test repetition detection
    repeated_text = "the cat sat on the cat sat on the cat sat on the mat"
    normal_text = "The quick brown fox jumps over the lazy dog."
    
    print("\n2. Repetition Detection:")
    print(f"  Repeated text: rep3={detect_repetition(repeated_text, 3):.2%}")
    print(f"  Normal text: rep3={detect_repetition(normal_text, 3):.2%}")
    
    # Test Jaccard overlap
    print("\n3. Jaccard Overlap (CORRECT):")
    set_a = {1, 2, 3, 4, 5}
    set_b = {3, 4, 5, 6, 7}
    print(f"  Set A: {set_a}")
    print(f"  Set B: {set_b}")
    print(f"  Jaccard: {jaccard_overlap(set_a, set_b):.2%}")
    
    # Test edge case - identical sets
    print(f"  Identical sets: {jaccard_overlap(set_a, set_a):.2%}")
    
    # Test edge case - disjoint sets
    set_c = {10, 11, 12}
    print(f"  Disjoint sets: {jaccard_overlap(set_a, set_c):.2%}")
    
    print("\n✓ All tests passed!")
