"""Comprehensive Evaluation Module for SAE Multilingual Steering.

Integrates:
1. Script detection with dominance margin (rejects code-mixed text)
2. Semantic similarity via LaBSE
3. LLM-as-Judge with CALIBRATION (Lee et al., 2025 - arXiv:2511.21140)
4. CORRECT Jaccard overlap (always ≤ 1.0)
5. Repetition/degradation metrics
6. Pareto frontier analysis

Key insight from Lee et al. (2025):
- Raw LLM judge scores are BIASED
- θ̂ = (p̂ + q₀ - 1) / (q₀ + q₁ - 1) corrects for bias
- Need calibration set with ground-truth labels
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List, Optional, Set, Tuple
from scipy.stats import norm

import numpy as np

from config import (
    SCRIPT_RANGES,
    SCRIPT_DETECTION_THRESHOLD,
    DEVANAGARI_THRESHOLD,
    REPETITION_3GRAM_THRESHOLD,
    REPETITION_5GRAM_THRESHOLD,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_SIM_THRESHOLD,
    GOOGLE_API_KEY,
)


###############################################################################
# SEMANTIC MODEL (LaBSE)
###############################################################################

_SEMANTIC_MODEL_CACHE: Dict[str, Any] = {}


def get_semantic_model(model_name: str = SEMANTIC_MODEL_NAME):
    """Lazy-load and cache a SentenceTransformer model."""
    if model_name in _SEMANTIC_MODEL_CACHE:
        return _SEMANTIC_MODEL_CACHE[model_name]
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _SEMANTIC_MODEL_CACHE[model_name] = model
        print(f"[eval] Loaded semantic model: {model_name}")
        return model
    except Exception as e:
        print(f"[eval] Warning: could not load semantic model: {e}")
        print("[eval] Install with: pip install sentence-transformers")
        return None


def semantic_similarity(text_a: str, text_b: str, model_name: str = SEMANTIC_MODEL_NAME) -> float:
    """Compute cosine similarity between two texts using LaBSE.
    
    Returns -1.0 if model unavailable.
    """
    model = get_semantic_model(model_name)
    if model is None:
        return -1.0
    
    embs = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SteeringEvalResult:
    """Result of evaluating a single steered generation."""
    prompt: str
    output: str
    method: str
    strength: float
    layer: int
    
    # Script metrics
    script_ratios: Dict[str, float]
    is_target_script: bool
    detected_language: str
    
    # Degradation
    repetition_3gram: float
    repetition_5gram: float
    is_degraded: bool
    
    # Semantic preservation
    semantic_similarity: float = -1.0
    semantics_ok: bool = False
    reference: str = ""
    
    # LLM-as-Judge (raw scores)
    llm_language_score: float = 0.0      # 1-5
    llm_faithfulness_score: float = 0.0  # 1-5
    llm_coherence_score: float = 0.0     # 1-5
    llm_judge_raw: Optional[Dict] = None
    
    # Combined success
    overall_success: bool = False


@dataclass
class CalibratedJudgeResult:
    """Result from calibrated LLM-as-Judge evaluation (Lee et al., 2025).
    
    Implements bias correction: θ̂ = (p̂ + q₀ - 1) / (q₀ + q₁ - 1)
    """
    raw_accuracy: float        # p̂: raw proportion judged correct
    corrected_accuracy: float  # θ̂: bias-adjusted estimate
    confidence_interval: Tuple[float, float]  # (lower, upper)
    
    q0: float  # Specificity: P(judge=incorrect | truly incorrect)
    q1: float  # Sensitivity: P(judge=correct | truly correct)
    
    n_test: int       # Test set size
    n_calib_0: int    # Calibration samples with true label 0
    n_calib_1: int    # Calibration samples with true label 1


@dataclass
class AggregateResults:
    """Aggregated results across multiple evaluations.

    Note: avg_target_script_ratio is defined with respect to a caller-specified
    target script (e.g. Devanagari for Hindi, Bengali for bn, etc.). Callers
    should pass the correct target_script when calling aggregate_results.
    """
    n_samples: int
    success_rate: float                    # Script-based (is_target_script)
    semantic_success_rate: Optional[float] # Script + semantic
    
    avg_target_script_ratio: float
    avg_semantic_similarity: Optional[float]
    avg_repetition_3gram: float
    avg_repetition_5gram: float
    degradation_rate: float
    
    # LLM judge (if calibrated)
    calibrated_result: Optional[CalibratedJudgeResult] = None
    
    # Per-strength breakdown
    per_strength: Dict[float, Dict] = field(default_factory=dict)


# =============================================================================
# SCRIPT DETECTION (IMPROVED - rejects code-mixed)
# =============================================================================

def detect_scripts(text: str) -> Dict[str, float]:
    """Detect script distribution over alphabetic characters."""
    if not text:
        return {}
    
    alpha_chars = [c for c in text if c.isalpha()]
    total_alpha = len(alpha_chars)
    if total_alpha == 0:
        return {}
    
    ratios = {}
    for script_name, (start, end) in SCRIPT_RANGES.items():
        count = sum(1 for c in text if start <= ord(c) <= end)
        ratios[script_name] = count / total_alpha
    
    return ratios


def is_target_script(
    text: str,
    target: str = "devanagari",
    threshold: float = DEVANAGARI_THRESHOLD,
    dominance_margin: float = 0.2,
) -> bool:
    """Check if text is PRIMARILY in target script.
    
    Rejects code-mixed text like "Hello नमस्ते" by requiring:
    1. Target ratio >= threshold
    2. Target ratio >= (next highest ratio + dominance_margin)
    """
    ratios = detect_scripts(text)
    if not ratios:
        return False
    
    target_ratio = ratios.get(target, 0.0)
    
    if target_ratio < threshold:
        return False
    
    # Dominance check: target must be clearly dominant
    max_other = max((r for s, r in ratios.items() if s != target), default=0.0)
    if target_ratio - max_other < dominance_margin:
        return False
    
    return True


def detect_language(text: str) -> Tuple[str, float]:
    """Detect primary language from script with confidence."""
    ratios = detect_scripts(text)
    if not ratios:
        return "unknown", 0.0
    
    primary_script = max(ratios, key=ratios.get)
    confidence = ratios[primary_script]
    
    script_to_lang = {
        "devanagari": "hi",
        "arabic": "ur",
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "gujarati": "gu",
        "gurmukhi": "pa",
        "oriya": "or",
        "han": "zh",
        "latin": "en",
    }
    
    lang = script_to_lang.get(primary_script, "unknown")
    
    # For Latin, try to distinguish languages
    if primary_script == "latin":
        lang, confidence = _detect_latin_language(text, confidence)
    
    return lang, confidence


def _detect_latin_language(text: str, base_conf: float) -> Tuple[str, float]:
    """Detect which Latin-script language."""
    text_lower = text.lower()
    
    patterns = {
        "de": r'\b(der|die|das|und|ist|ein|nicht|sie|ich|mit)\b',
        "es": r'\b(el|la|los|las|es|una|que|por|para|con)\b',
        "fr": r'\b(le|la|les|de|est|un|une|que|pour|dans)\b',
        "en": r'\b(the|is|are|was|were|have|has|with|this|that)\b',
    }
    
    scores = {lang: len(re.findall(p, text_lower)) for lang, p in patterns.items()}
    
    if max(scores.values()) == 0:
        return "en", base_conf * 0.5
    
    best = max(scores, key=scores.get)
    return best, base_conf * min(1.0, scores[best] / 5)


def is_code_mixed(text: str, threshold: float = 0.2) -> bool:
    """Check if text is code-mixed (multiple scripts present)."""
    ratios = detect_scripts(text)
    significant = [s for s, r in ratios.items() if r > threshold]
    return len(significant) > 1


# =============================================================================
# REPETITION / DEGRADATION
# =============================================================================

def compute_ngram_repetition(text: str, n: int) -> float:
    """Compute n-gram repetition rate."""
    if not text or len(text) < n:
        return 0.0
    
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    
    counts = Counter(ngrams)
    return (len(ngrams) - len(counts)) / len(ngrams)


def is_degraded(
    text: str,
    threshold_3gram: float = REPETITION_3GRAM_THRESHOLD,
    threshold_5gram: float = REPETITION_5GRAM_THRESHOLD,
) -> bool:
    """Check if text shows degradation."""
    return (compute_ngram_repetition(text, 3) > threshold_3gram or 
            compute_ngram_repetition(text, 5) > threshold_5gram)


# =============================================================================
# JACCARD OVERLAP (CORRECT!)
# =============================================================================

def jaccard_overlap(set_a: Set, set_b: Set) -> float:
    """CORRECT Jaccard overlap: |A∩B| / |A∪B|, always in [0, 1]."""
    if not set_a and not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    result = intersection / union
    assert 0.0 <= result <= 1.0, f"Invalid Jaccard: {result}"
    return result


def dice_overlap(set_a: Set, set_b: Set) -> float:
    """Dice coefficient: 2|A∩B| / (|A| + |B|)."""
    if not set_a and not set_b:
        return 0.0
    return 2 * len(set_a & set_b) / (len(set_a) + len(set_b))


###############################################################################
# LLM-AS-JUDGE WITH CALIBRATION (Lee et al., 2025)
###############################################################################

# Mapping from language code to human-readable description used in the
# judge prompt. This keeps the prompt grounded for each language while
# letting callers reuse the same underlying judge function.
LANG_JUDGE_CONFIG = {
    "hi": {"name": "Hindi", "script": "Devanagari"},
    "ur": {"name": "Urdu", "script": "Arabic script"},
    "bn": {"name": "Bengali", "script": "Bengali script"},
    "ta": {"name": "Tamil", "script": "Tamil script"},
    "te": {"name": "Telugu", "script": "Telugu script"},
    "de": {"name": "German", "script": "Latin alphabet"},
    "ar": {"name": "Arabic", "script": "Arabic script"},
}


JUDGE_PROMPT = """You are evaluating a language model output for steering into a target language.

Target language: {lang_name} ({script_name})

Original prompt:
{prompt}

Model output:
{output}

Score EACH on a scale of 1–5:

1. **Language** (1 = not {lang_name} / wrong script,
                 3 = mixed or broken {lang_name},
                 5 = fluent {lang_name} in {script_name})
2. **Faithfulness** (1 = unrelated to prompt, 5 = fully answers the prompt)
3. **Coherence** (1 = nonsense or highly repetitive, 5 = coherent and readable)

Respond with ONLY JSON (no markdown), e.g.:
{{"language": 1-5, "faithfulness": 1-5, "coherence": 1-5,
  "is_target_language": true/false, "brief": "one sentence justification"}}"""


_GEMINI_AVAILABLE: Optional[bool] = None


def is_gemini_available() -> bool:
    """Lightweight runtime check that Gemini API is usable.

    We check once per process and cache the result. This avoids the pattern
    where an experiment silently runs with `use_llm_judge=True` but every
    call fails due to a missing/invalid API key.
    """
    global _GEMINI_AVAILABLE
    if _GEMINI_AVAILABLE is not None:
        return _GEMINI_AVAILABLE

    api_key = GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[eval] No Gemini API key found (GOOGLE_API_KEY / GEMINI_API_KEY). LLM judge disabled.")
        _GEMINI_AVAILABLE = False
        return False

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        # Use the latest fast Gemini model for judging.
        model = genai.GenerativeModel("gemini-2.5-flash")
        # Minimal sanity call; we don't care about the content, only that it
        # does not raise.
        _ = model.generate_content("Ping from SAE-Multilingual-Steering evaluation.")
        _GEMINI_AVAILABLE = True
        print("[eval] Gemini API check succeeded; LLM judge enabled.")
        return True
    except Exception as e:
        print(f"[eval] Gemini API check failed: {e}")
        _GEMINI_AVAILABLE = False
        return False


def llm_judge_gemini(
    prompt: str,
    output: str,
    api_key: str = None,
    lang_code: str = "hi",
) -> Optional[Dict]:
    """Use Gemini as judge. Returns scores or None on failure.

    This function assumes `is_gemini_available()` has been called at least
    once in the current process. If no API key is available or the check
    failed, this function will simply return None.
    """
    api_key = (
        api_key
        or GOOGLE_API_KEY
        or os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )

    if not api_key or _GEMINI_AVAILABLE is False:
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        cfg = LANG_JUDGE_CONFIG.get(lang_code, LANG_JUDGE_CONFIG["hi"])
        response = model.generate_content(
            JUDGE_PROMPT.format(
                prompt=prompt,
                output=output,
                lang_name=cfg["name"],
                script_name=cfg["script"],
            )
        )
        text = response.text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"```\w*\n?", "", text).strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        return json.loads(text)

    except Exception as e:
        print(f"[eval] LLM judge error: {e}")
        return None


def _clip(x: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clip value to [low, high]."""
    return max(low, min(high, x))


def bias_adjusted_estimator(p: float, q0: float, q1: float) -> float:
    """Compute bias-adjusted accuracy estimate (Lee et al., 2025 Eq. 4).
    
    θ̂ = (p̂ + q₀ - 1) / (q₀ + q₁ - 1)
    
    Args:
        p: Raw proportion judged correct
        q0: Specificity (judge accuracy on truly incorrect)
        q1: Sensitivity (judge accuracy on truly correct)
    """
    if q0 + q1 <= 1:
        return p  # Can't correct if judge is worse than random
    
    theta = (p + q0 - 1) / (q0 + q1 - 1)
    return _clip(theta)


def confidence_interval(
    p: float, q0: float, q1: float,
    n: int, m0: int, m1: int,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute confidence interval for θ (Lee et al., 2025 Eq. 5-7).
    
    Args:
        p: Raw proportion (test set)
        q0, q1: Specificity, sensitivity (from calibration)
        n: Test set size
        m0: Calibration samples with true label 0 (incorrect)
        m1: Calibration samples with true label 1 (correct)
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        (lower, upper) confidence interval
    """
    z = norm.ppf(1 - alpha / 2)  # e.g., 1.96 for 95%
    
    # Adjusted quantities (Eq. 6)
    n_tilde = n + z**2
    m0_tilde = m0 + 2
    m1_tilde = m1 + 2
    
    p_tilde = (n * p + z**2 / 2) / n_tilde
    q0_tilde = (m0 * q0 + 1) / m0_tilde
    q1_tilde = (m1 * q1 + 1) / m1_tilde
    
    # Adjusted theta (Eq. 7)
    if q0_tilde + q1_tilde <= 1:
        return (0.0, 1.0)
    
    theta_tilde = (p_tilde + q0_tilde - 1) / (q0_tilde + q1_tilde - 1)
    
    d_theta = 2 * z**2 * (
        -(1 - theta_tilde) * q0_tilde * (1 - q0_tilde) / m0_tilde
        + theta_tilde * q1_tilde * (1 - q1_tilde) / m1_tilde
    )
    
    # Standard error (Eq. 5)
    var_term = (
        p_tilde * (1 - p_tilde) / n_tilde
        + (1 - theta_tilde)**2 * q0_tilde * (1 - q0_tilde) / m0_tilde
        + theta_tilde**2 * q1_tilde * (1 - q1_tilde) / m1_tilde
    )
    se = sqrt(var_term) / (q0_tilde + q1_tilde - 1)
    
    # Confidence interval
    lower = _clip(theta_tilde + d_theta - z * se)
    upper = _clip(theta_tilde + d_theta + z * se)
    
    return (lower, upper)


def calibrate_judge(
    judge_predictions: List[bool],
    ground_truth: List[bool],
) -> Tuple[float, float]:
    """Estimate judge specificity (q0) and sensitivity (q1) from calibration set.
    
    Args:
        judge_predictions: List of judge decisions (True = "correct")
        ground_truth: List of actual labels (True = truly correct)
    
    Returns:
        (q0, q1): Specificity, Sensitivity
    """
    if len(judge_predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    # Count by true label
    true_positives = sum(1 for p, g in zip(judge_predictions, ground_truth) if p and g)
    true_negatives = sum(1 for p, g in zip(judge_predictions, ground_truth) if not p and not g)
    
    n_truly_correct = sum(ground_truth)
    n_truly_incorrect = len(ground_truth) - n_truly_correct
    
    q1 = true_positives / n_truly_correct if n_truly_correct > 0 else 0.5
    q0 = true_negatives / n_truly_incorrect if n_truly_incorrect > 0 else 0.5
    
    return q0, q1


def evaluate_with_calibrated_judge(
    test_outputs: List[Dict],  # [{prompt, output, ...}]
    calibration_set: List[Dict],  # [{prompt, output, ground_truth_is_correct}]
    judge_fn=llm_judge_gemini,
    success_threshold: int = 3,  # Language score >= 3 = success
) -> CalibratedJudgeResult:
    """Full evaluation with calibrated LLM-as-judge.
    
    This implements the method from Lee et al. (2025):
    1. Run judge on calibration set to estimate q0, q1
    2. Run judge on test set to get raw p̂
    3. Compute bias-adjusted θ̂ and confidence interval
    """
    print(f"[eval] Running calibrated judge evaluation...")
    print(f"[eval] Calibration set: {len(calibration_set)}, Test set: {len(test_outputs)}")
    
    # Step 1: Calibrate judge
    calib_preds = []
    calib_truths = []
    
    for item in calibration_set:
        result = judge_fn(item["prompt"], item["output"])
        if result is None:
            continue
        
        judge_says_correct = result.get("language", 0) >= success_threshold
        calib_preds.append(judge_says_correct)
        calib_truths.append(item["ground_truth_is_correct"])
    
    if len(calib_preds) < 10:
        print("[eval] Warning: Insufficient calibration data")
        return None
    
    q0, q1 = calibrate_judge(calib_preds, calib_truths)
    print(f"[eval] Calibration: q0={q0:.3f} (specificity), q1={q1:.3f} (sensitivity)")
    
    # Step 2: Run on test set
    test_preds = []
    for item in test_outputs:
        result = judge_fn(item["prompt"], item["output"])
        if result is None:
            continue
        judge_says_correct = result.get("language", 0) >= success_threshold
        test_preds.append(judge_says_correct)
    
    if not test_preds:
        return None
    
    # Raw proportion
    p_hat = sum(test_preds) / len(test_preds)
    
    # Step 3: Bias correction
    theta_hat = bias_adjusted_estimator(p_hat, q0, q1)
    
    # Step 4: Confidence interval
    n_calib_0 = sum(1 for t in calib_truths if not t)
    n_calib_1 = sum(1 for t in calib_truths if t)
    
    ci = confidence_interval(p_hat, q0, q1, len(test_preds), n_calib_0, n_calib_1)
    
    print(f"[eval] Raw accuracy: {p_hat:.3f}")
    print(f"[eval] Corrected accuracy: {theta_hat:.3f}")
    print(f"[eval] 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    return CalibratedJudgeResult(
        raw_accuracy=p_hat,
        corrected_accuracy=theta_hat,
        confidence_interval=ci,
        q0=q0,
        q1=q1,
        n_test=len(test_preds),
        n_calib_0=n_calib_0,
        n_calib_1=n_calib_1,
    )


# =============================================================================
# CALIBRATION TABLE HELPERS (re-use q0, q1 across experiments)
# =============================================================================

_JUDGE_CALIBRATION_TABLE: Optional[Dict[str, Dict[str, float]]] = None


def load_judge_calibration_table(
    path: str = "results/exp11_judge_calibration.json",
) -> Dict[str, Dict[str, float]]:
    """Load per-language judge calibration stats from Exp11.

    Returns a mapping:
        lang -> {q0, q1, n_calib_0, n_calib_1}

    If the file is missing or malformed, returns an empty dict.
    """
    global _JUDGE_CALIBRATION_TABLE
    if _JUDGE_CALIBRATION_TABLE is not None:
        return _JUDGE_CALIBRATION_TABLE

    table: Dict[str, Dict[str, float]] = {}
    try:
        with open(path) as f:
            data = json.load(f)
        langs = data.get("languages", {})
        for lang, stats in langs.items():
            if not isinstance(stats, dict):
                continue
            q0 = stats.get("q0")
            q1 = stats.get("q1")
            n0 = stats.get("n_calib_0")
            n1 = stats.get("n_calib_1")
            if None in (q0, q1, n0, n1):
                continue
            table[lang] = {
                "q0": float(q0),
                "q1": float(q1),
                "n_calib_0": int(n0),
                "n_calib_1": int(n1),
            }
    except FileNotFoundError:
        print("[eval] Judge calibration file not found; calibrated summaries will be unavailable.")
    except Exception as e:
        print(f"[eval] Error loading judge calibration file: {e}")

    _JUDGE_CALIBRATION_TABLE = table
    return table


def calibrated_judge_from_results(
    results: List[SteeringEvalResult],
    lang: str,
    calibration_table: Dict[str, Dict[str, float]],
    success_threshold: int = 3,
) -> Optional[CalibratedJudgeResult]:
    """Compute calibrated judge accuracy for a set of results.

    This reuses (q0, q1, n_calib_0, n_calib_1) estimated once in Exp11 for
    `lang` and applies bias_adjusted_estimator / confidence_interval on the
    judge decisions stored in the provided SteeringEvalResults.
    """
    if not results:
        return None

    stats = calibration_table.get(lang)
    if not stats:
        return None

    q0 = stats["q0"]
    q1 = stats["q1"]
    m0 = stats["n_calib_0"]
    m1 = stats["n_calib_1"]

    preds: List[bool] = []
    for r in results:
        # If no judge result was recorded, skip this sample.
        if r.llm_judge_raw is None:
            continue
        preds.append(r.llm_language_score >= success_threshold)

    if not preds:
        return None

    n = len(preds)
    p_hat = sum(preds) / n
    theta_hat = bias_adjusted_estimator(p_hat, q0, q1)
    ci = confidence_interval(p_hat, q0, q1, n, m0, m1)

    return CalibratedJudgeResult(
        raw_accuracy=p_hat,
        corrected_accuracy=theta_hat,
        confidence_interval=ci,
        q0=q0,
        q1=q1,
        n_test=n,
        n_calib_0=m0,
        n_calib_1=m1,
    )


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_steering_output(
    prompt: str,
    output: str,
    method: str = "unknown",
    strength: float = 0.0,
    layer: int = 0,
    reference: str = None,
    target_script: str = "devanagari",
    use_llm_judge: bool = False,
    compute_semantics: bool = True,
    judge_lang: Optional[str] = None,
) -> SteeringEvalResult:
    """Comprehensive evaluation of a single steered output."""
    
    # Script detection
    script_ratios = detect_scripts(output)
    is_target = is_target_script(output, target=target_script)
    lang, lang_conf = detect_language(output)
    
    # Degradation
    rep_3 = compute_ngram_repetition(output, 3)
    rep_5 = compute_ngram_repetition(output, 5)
    degraded = is_degraded(output)
    
    # Semantic similarity
    sem_sim = -1.0
    sem_ok = False
    ref_text = reference or prompt
    
    if compute_semantics and ref_text:
        sem_sim = semantic_similarity(ref_text, output)
        if sem_sim >= 0:
            sem_ok = sem_sim >= SEMANTIC_SIM_THRESHOLD
    
    # LLM judge
    llm_lang = llm_faith = llm_coh = 0.0
    llm_raw: Optional[Any] = None
    
    if use_llm_judge:
        raw = llm_judge_gemini(prompt, output, lang_code=judge_lang or "hi")
        # Gemini sometimes returns a top-level list or other structures;
        # we defensively convert to a single dict if possible.
        if isinstance(raw, list) and raw:
            candidate = raw[0]
        elif isinstance(raw, dict):
            candidate = raw
        else:
            candidate = None
        llm_raw = raw

        if isinstance(candidate, dict):
            llm_lang = candidate.get("language", 0)
            llm_faith = candidate.get("faithfulness", 0)
            llm_coh = candidate.get("coherence", 0)
    
    # Overall success: target script + not degraded + semantics OK (if computed)
    success = is_target and not degraded
    if compute_semantics and sem_sim >= 0:
        success = success and sem_ok
    
    return SteeringEvalResult(
        prompt=prompt,
        output=output,
        method=method,
        strength=strength,
        layer=layer,
        script_ratios=script_ratios,
        is_target_script=is_target,
        detected_language=lang,
        repetition_3gram=rep_3,
        repetition_5gram=rep_5,
        is_degraded=degraded,
        semantic_similarity=sem_sim,
        semantics_ok=sem_ok,
        reference=ref_text,
        llm_language_score=llm_lang,
        llm_faithfulness_score=llm_faith,
        llm_coherence_score=llm_coh,
        llm_judge_raw=llm_raw,
        overall_success=success,
    )


def aggregate_results(
    results: List[SteeringEvalResult],
    target_script: str = "devanagari",
) -> AggregateResults:
    """Aggregate evaluation results.

    Args:
        results: List of SteeringEvalResult objects.
        target_script: Script name to use when computing
            avg_target_script_ratio. This should match the target language
            for the steering experiment, e.g. "devanagari" for Hindi,
            "bengali" for Bengali, "tamil" for Tamil, etc.
    """
    if not results:
        return AggregateResults(
            n_samples=0, success_rate=0.0, semantic_success_rate=None,
            avg_target_script_ratio=0.0, avg_semantic_similarity=None,
            avg_repetition_3gram=0.0, avg_repetition_5gram=0.0, degradation_rate=0.0
        )
    
    n = len(results)
    
    # Script success
    script_success = sum(1 for r in results if r.is_target_script) / n
    
    # Target script ratio
    target_ratios = [r.script_ratios.get(target_script, 0.0) for r in results]
    
    # Semantic
    sem_values = [r.semantic_similarity for r in results if r.semantic_similarity >= 0]
    avg_sem = np.mean(sem_values) if sem_values else None
    
    # Semantic + script success
    sem_success = None
    if sem_values:
        sem_success_count = sum(1 for r in results if r.overall_success)
        sem_success = sem_success_count / n
    
    # Degradation
    deg_rate = sum(1 for r in results if r.is_degraded) / n
    
    return AggregateResults(
        n_samples=n,
        success_rate=script_success,
        semantic_success_rate=sem_success,
        avg_target_script_ratio=np.mean(target_ratios),
        avg_semantic_similarity=avg_sem,
        avg_repetition_3gram=np.mean([r.repetition_3gram for r in results]),
        avg_repetition_5gram=np.mean([r.repetition_5gram for r in results]),
        degradation_rate=deg_rate,
    )


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Testing comprehensive evaluation module...\n")
    
    # 1. Jaccard tests
    print("1. Jaccard overlap (must always be ≤ 1.0)...")
    test_cases = [
        ({1, 2, 3, 4, 5}, {3, 4, 5, 6, 7}, 3/7),
        ({1, 2, 3}, {1, 2, 3}, 1.0),
        ({1, 2}, {3, 4}, 0.0),
        (set(), set(), 0.0),
        (set(range(100)), set(range(50, 150)), 50/150),
    ]
    
    for a, b, expected in test_cases:
        j = jaccard_overlap(a, b)
        assert 0.0 <= j <= 1.0, f"Jaccard out of range: {j}"
        assert abs(j - expected) < 0.001, f"Jaccard wrong: {j} != {expected}"
    print("   ✓ All Jaccard tests passed (no >100% bug!)\n")
    
    # 2. Script detection
    print("2. Script detection...")
    assert is_target_script("नमस्ते दुनिया"), "Pure Hindi should pass"
    assert not is_target_script("Hello world"), "English should fail"
    assert not is_target_script("Hello नमस्ते"), "Mixed should fail (dominance check)"
    print("   ✓ Script detection passed (code-mixed rejected)\n")
    
    # 3. Bias correction
    print("3. LLM-as-Judge bias correction (Lee et al., 2025)...")
    # Example: q0=0.7, q1=0.9 (imperfect judge)
    p_raw = 0.5
    theta = bias_adjusted_estimator(p_raw, q0=0.7, q1=0.9)
    print(f"   Raw: {p_raw:.2f}, Corrected: {theta:.2f}")
    
    ci = confidence_interval(p_raw, 0.7, 0.9, n=100, m0=50, m1=50)
    print(f"   95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print("   ✓ Bias correction implemented\n")
    
    print("✓ All tests passed!")
