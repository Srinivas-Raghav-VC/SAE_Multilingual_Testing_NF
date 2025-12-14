"""Comprehensive Evaluation Module for SAE Multilingual Steering.

Integrates:
1. Script detection with dominance margin (rejects code-mixed text)
2. Semantic similarity via LaBSE
3. LLM-as-Judge with calibration (bias correction)
4. CORRECT Jaccard overlap (always ≤ 1.0)
5. Repetition/degradation metrics
6. Pareto frontier analysis

Key insight:
- Raw LLM judge scores can be biased.
- We correct observed rates using a sensitivity/specificity calibration
  estimated from a labeled calibration set.
"""

import json
import os
import re
import hashlib
import random
import time
from pathlib import Path
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
    MIN_JUDGE_CALIB_PER_CLASS,
)

import unicodedata

# =============================================================================
# CONSTANTS
# =============================================================================

LID_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for reliable language detection


###############################################################################
# SEMANTIC MODEL (LaBSE)
###############################################################################

_SEMANTIC_MODEL_CACHE: Dict[str, Any] = {}
_TRUNCATION_WARNED: bool = False
# Track truncation frequency for reporting/debugging.
_SEMANTIC_CALLS: int = 0
_SEMANTIC_TRUNCATED: int = 0


###############################################################################
# LLM JUDGE (Gemini) + RATE LIMITING/CACHE
###############################################################################

_LLM_JUDGE_CACHE: Dict[str, Dict] = {}
_LLM_JUDGE_CACHE_PATH = Path("results/llm_judge_cache.jsonl")
_LLM_JUDGE_CACHE_LOADED: bool = False

_LAST_JUDGE_TS: float = 0.0
_JUDGE_WINDOW_START: float = 0.0
_JUDGE_CALLS_IN_WINDOW: int = 0


def _load_llm_judge_cache() -> None:
    """Load on-disk judge cache once per process."""
    global _LLM_JUDGE_CACHE_LOADED
    if _LLM_JUDGE_CACHE_LOADED:
        return
    _LLM_JUDGE_CACHE_LOADED = True
    if not _LLM_JUDGE_CACHE_PATH.exists():
        return
    try:
        with _LLM_JUDGE_CACHE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                resp = obj.get("response")
                if key and isinstance(resp, dict):
                    _LLM_JUDGE_CACHE[key] = resp
    except Exception as e:
        print(f"[eval] Warning: could not load LLM judge cache: {e}")


def _judge_cache_key(prompt: str, output: str, lang_code: str) -> str:
    h = hashlib.sha256()
    h.update(lang_code.encode("utf-8"))
    h.update(b"\n---\n")
    h.update(prompt.encode("utf-8"))
    h.update(b"\n---\n")
    h.update(output.encode("utf-8"))
    return h.hexdigest()


def _respect_gemini_rpm(max_rpm: int) -> None:
    """Simple per-process RPM throttle."""
    global _LAST_JUDGE_TS, _JUDGE_WINDOW_START, _JUDGE_CALLS_IN_WINDOW
    now = time.time()
    if _JUDGE_WINDOW_START == 0.0 or (now - _JUDGE_WINDOW_START) >= 60.0:
        _JUDGE_WINDOW_START = now
        _JUDGE_CALLS_IN_WINDOW = 0

    # Smooth spacing between calls.
    if max_rpm > 0:
        min_interval = 60.0 / float(max_rpm)
        dt = now - _LAST_JUDGE_TS
        if dt < min_interval:
            time.sleep(min_interval - dt)

    _JUDGE_CALLS_IN_WINDOW += 1
    if max_rpm > 0 and _JUDGE_CALLS_IN_WINDOW > max_rpm:
        sleep_for = 60.0 - (now - _JUDGE_WINDOW_START) + 0.1
        print(f"[eval] Gemini RPM limit reached ({max_rpm}/min); sleeping {sleep_for:.1f}s.")
        time.sleep(max(sleep_for, 0.0))
        _JUDGE_WINDOW_START = time.time()
        _JUDGE_CALLS_IN_WINDOW = 1

    _LAST_JUDGE_TS = time.time()


def get_semantic_model(model_name: str = SEMANTIC_MODEL_NAME):
    """Lazy-load and cache a SentenceTransformer model.

    Defaulting this to GPU can cause OOM in shared-GPU setups because the main
    LLM already occupies most VRAM. We therefore default to CPU and allow
    opting into CUDA explicitly via ``SEMANTIC_DEVICE=cuda``.
    """
    device = os.environ.get("SEMANTIC_DEVICE", "cpu").strip().lower() or "cpu"
    cache_key = f"{model_name}::{device}"
    if cache_key in _SEMANTIC_MODEL_CACHE:
        return _SEMANTIC_MODEL_CACHE[cache_key]
    
    try:
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(model_name, device=device)
        except TypeError:
            model = SentenceTransformer(model_name)
            try:
                model = model.to(device)
            except Exception:
                pass
        _SEMANTIC_MODEL_CACHE[cache_key] = model
        print(f"[eval] Loaded semantic model: {model_name} (device={device})")
        return model
    except Exception as e:
        print(f"[eval] Warning: could not load semantic model: {e}")
        print("[eval] Install with: pip install sentence-transformers")
        return None


def _truncate_for_semantic_model(text: str, max_tokens: int = 512) -> str:
    """Best-effort truncation helper for LaBSE / sentence-transformers.

    The underlying LaBSE encoder typically has a 512-token limit; we
    explicitly truncate longer texts to avoid silent truncation inside
    the model. We approximate tokens with whitespace-separated words.
    """
    global _TRUNCATION_WARNED
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    global _SEMANTIC_TRUNCATED
    _SEMANTIC_TRUNCATED += 1
    # Warn once per process to avoid log spam.
    if not _TRUNCATION_WARNED:
        print(f"[eval] Warning: truncating text from {len(tokens)} to {max_tokens} tokens for semantic similarity.")
        _TRUNCATION_WARNED = True
    return " ".join(tokens[:max_tokens])


def semantic_similarity(text_a: str, text_b: str, model_name: str = SEMANTIC_MODEL_NAME) -> float:
    """Compute cosine similarity between two texts using LaBSE.
    
    Returns -1.0 if model unavailable.
    """
    global _SEMANTIC_CALLS
    _SEMANTIC_CALLS += 1
    model = get_semantic_model(model_name)
    if model is None:
        return -1.0

    text_a = _truncate_for_semantic_model(text_a)
    text_b = _truncate_for_semantic_model(text_b)
    embs = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def get_semantic_truncation_stats() -> Dict[str, int]:
    """Return process-level truncation counters for LaBSE calls."""
    return {"calls": _SEMANTIC_CALLS, "truncated_texts": _SEMANTIC_TRUNCATED}

###############################################################################
# QA METRICS (Exact Match / F1)
###############################################################################

def normalize_qa_text(text: str) -> str:
    """Language-agnostic QA normalization.

    We avoid English-specific heuristics (e.g., article removal) because this
    project evaluates multilingual QA. We:
      - lowercase,
      - strip Unicode punctuation,
      - normalize whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return " ".join(text.split())


def qa_exact_match(prediction: str, references: List[str]) -> float:
    """Exact match, max over references."""
    if not references:
        return 0.0
    pred = normalize_qa_text(prediction)
    return float(any(pred == normalize_qa_text(r) for r in references))


def qa_f1(prediction: str, references: List[str]) -> float:
    """Token F1, max over references."""
    if not references:
        return 0.0

    def _f1_single(pred: str, ref: str) -> float:
        pred_toks = normalize_qa_text(pred).split()
        ref_toks = normalize_qa_text(ref).split()
        if not pred_toks and not ref_toks:
            return 1.0
        if not pred_toks or not ref_toks:
            return 0.0

        common = Counter(pred_toks) & Counter(ref_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_toks)
        recall = num_same / len(ref_toks)
        return 2 * precision * recall / (precision + recall)

    return float(max(_f1_single(prediction, r) for r in references))


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

    # QA metrics (only set when a gold reference answer is provided)
    qa_exact_match: Optional[float] = None  # 0/1
    qa_f1: Optional[float] = None           # 0..1
    
    # LLM-as-Judge (raw scores)
    llm_language_score: float = 0.0      # 1-5
    llm_faithfulness_score: float = 0.0  # 1-5
    llm_coherence_score: float = 0.0     # 1-5
    llm_judge_raw: Optional[Dict] = None
    
    # Combined success
    overall_success: bool = False


@dataclass
class CalibratedJudgeResult:
    """Result from calibrated LLM-as-Judge evaluation.
    
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
class SeparateMetrics:
    """Individual metrics reported SEPARATELY (not combined).

    This addresses the reviewer concern about conflating different success criteria.
    Each metric should be analyzed independently to understand trade-offs.
    """
    # Script metrics (binary + continuous)
    script_success_rate: float          # Fraction with is_target_script=True
    script_success_ci: Tuple[float, float]  # 95% bootstrap CI
    avg_script_ratio: float             # Mean target script ratio
    script_ratio_ci: Tuple[float, float]

    # Semantic preservation (continuous)
    semantic_mean: Optional[float]      # Mean LaBSE similarity
    semantic_ci: Optional[Tuple[float, float]]
    semantic_preservation_rate: Optional[float]  # Fraction >= threshold

    # Degradation (binary + continuous)
    degradation_rate: float             # Fraction with is_degraded=True
    degradation_ci: Tuple[float, float]
    avg_repetition_3gram: float
    avg_repetition_5gram: float

    # Combined (for reference, but NOT primary metric)
    combined_success_rate: float        # Script AND semantic AND not-degraded

    def to_dict(self) -> Dict:
        return {
            "script_success_rate": self.script_success_rate,
            "script_success_ci": list(self.script_success_ci),
            "avg_script_ratio": self.avg_script_ratio,
            "script_ratio_ci": list(self.script_ratio_ci),
            "semantic_mean": self.semantic_mean,
            "semantic_ci": list(self.semantic_ci) if self.semantic_ci else None,
            "semantic_preservation_rate": self.semantic_preservation_rate,
            "degradation_rate": self.degradation_rate,
            "degradation_ci": list(self.degradation_ci),
            "avg_repetition_3gram": self.avg_repetition_3gram,
            "avg_repetition_5gram": self.avg_repetition_5gram,
            "combined_success_rate": self.combined_success_rate,
        }


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
    # Number of outputs where script detection found no alphabetic
    # characters (possible degenerate generations).
    n_empty_outputs: int = 0

    # LLM judge (if calibrated)
    calibrated_result: Optional[CalibratedJudgeResult] = None

    # Per-strength breakdown
    per_strength: Dict[float, Dict] = field(default_factory=dict)

    # NEW: Separate metrics with CIs (for rigorous reporting)
    separate_metrics: Optional[SeparateMetrics] = None

    # NEW: Per-prompt raw values for downstream statistical testing
    per_prompt_script_success: Optional[List[int]] = None  # 0/1 per prompt
    per_prompt_semantic_sim: Optional[List[float]] = None
    per_prompt_degraded: Optional[List[int]] = None  # 0/1 per prompt

    # NEW: Sensitivity analysis results (threshold sweeps)
    sensitivity: Optional[List[Dict[str, float]]] = None


# =============================================================================
# SCRIPT DETECTION (IMPROVED - rejects code-mixed)
# =============================================================================

def _char_in_ranges(char: str, ranges) -> bool:
    """Check if character falls within any of the given ranges.

    Args:
        char: Single character
        ranges: Either a tuple (start, end) or a list of tuples [(start1, end1), ...]

    Returns:
        True if char falls within any range
    """
    code = ord(char)

    # Handle legacy single-range format: (start, end)
    if isinstance(ranges, tuple) and len(ranges) == 2 and isinstance(ranges[0], int):
        return ranges[0] <= code <= ranges[1]

    # Handle new multi-range format: [(start1, end1), (start2, end2), ...]
    if isinstance(ranges, list):
        for start, end in ranges:
            if start <= code <= end:
                return True
        return False

    return False


def detect_scripts(text: str) -> Dict[str, float]:
    """Detect script distribution over alphabetic characters.

    Supports both legacy single-range and new multi-range script definitions.
    Uses Unicode category checking for robust alpha detection.
    """
    if not text:
        return {}

    # Use unicodedata for robust alphabetic character detection
    # This handles all Unicode alphabetic characters correctly
    import unicodedata
    alpha_chars = [c for c in text if unicodedata.category(c).startswith('L')]
    total_alpha = len(alpha_chars)

    if total_alpha == 0:
        return {}

    ratios = {}
    # Count only alphabetic characters for both numerator and denominator.
    # This avoids ratios > 1.0 caused by counting combining marks (Mn/Mc)
    # that are in script ranges but not category 'L'.
    for script_name, ranges in SCRIPT_RANGES.items():
        count = sum(1 for c in alpha_chars if _char_in_ranges(c, ranges))
        ratios[script_name] = count / total_alpha

    return ratios


def detect_script(text: str) -> str:
    """Get the dominant script in text (compatibility wrapper).

    Returns the script name with highest ratio, or 'unknown' if no scripts detected.
    For more detailed analysis, use detect_scripts() which returns all ratios.
    """
    ratios = detect_scripts(text)
    if not ratios:
        return "unknown"
    return max(ratios, key=ratios.get)


# Compatibility alias for experiments that use the old name
compute_semantic_similarity = semantic_similarity


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
    # For Arabic script, use script-aware disambiguation
    elif primary_script == "arabic":
        lang, confidence = _detect_arabic_language(text, confidence)
    
    return lang, confidence


def is_reliable_language_detection(text: str) -> Tuple[bool, str, float]:
    """Check if detected language is reliable (confidence >= threshold).

    Returns:
        (is_reliable, detected_lang, confidence)
    """
    lang, conf = detect_language(text)
    return conf >= LID_CONFIDENCE_THRESHOLD, lang, conf


def get_language_with_confidence(text: str) -> Tuple[str, float, bool]:
    """Get detected language with confidence and reliability flag.

    This is the primary API for experiments needing LID with quality metrics.

    Returns:
        (language_code, confidence, is_reliable)
    """
    lang, conf = detect_language(text)
    reliable = conf >= LID_CONFIDENCE_THRESHOLD
    return lang, conf, reliable


_FASTTEXT_LID_CONFIG: Optional[Any] = None


def _fasttext_langdetect(text: str, allowed_langs: Optional[Set[str]] = None) -> Optional[Tuple[str, float]]:
    """FastText-based language identification (optional).

    Uses the `fast-langdetect` package (recommended) which wraps fastText LID
    models without requiring compilation on many platforms.

    Env vars:
      - LID_BACKEND=fasttext|auto|langid|regex
      - FASTTEXT_MODEL=lite|full (default: full)
      - FASTTEXT_LID_MODEL_PATH=/path/to/lid.176.ftz|lid.176.bin (optional)
      - FASTTEXT_CACHE_DIR=/path/to/cache (optional)
      - FASTTEXT_DISABLE_VERIFY=1 (optional)
    """
    global _FASTTEXT_LID_CONFIG
    try:
        from fast_langdetect import detect as ft_detect  # type: ignore
        from fast_langdetect import LangDetectConfig  # type: ignore
    except Exception:
        return None

    # Model choice also influences which local default model we prefer.
    model_choice = os.environ.get("FASTTEXT_MODEL", "full").strip().lower()
    if model_choice not in ("full", "lite"):
        model_choice = "full"

    if _FASTTEXT_LID_CONFIG is None:
        model_path = os.environ.get("FASTTEXT_LID_MODEL_PATH", "").strip()
        if not model_path:
            # Prefer local models/ when available to avoid network downloads at
            # runtime. This is also reviewer-friendly because it makes LID
            # deterministic across runs.
            base_dir = Path(__file__).resolve().parent / "models"
            preferred = base_dir / ("lid.176.bin" if model_choice == "full" else "lid.176.ftz")
            fallback = base_dir / ("lid.176.ftz" if model_choice == "full" else "lid.176.bin")
            if preferred.exists():
                model_path = str(preferred)
            elif fallback.exists():
                model_path = str(fallback)

        if model_path and not Path(model_path).exists():
            print(
                f"[eval] Warning: FASTTEXT_LID_MODEL_PATH='{model_path}' does not exist; "
                "falling back to fast-langdetect defaults."
            )
            model_path = ""

        cache_dir = os.environ.get("FASTTEXT_CACHE_DIR", "").strip()
        disable_verify = str(os.environ.get("FASTTEXT_DISABLE_VERIFY", "0")).lower() in ("1", "true", "yes")
        kwargs: Dict[str, Any] = {}
        if model_path:
            kwargs["custom_model_path"] = model_path
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if disable_verify:
            kwargs["disable_verify"] = True
        try:
            _FASTTEXT_LID_CONFIG = LangDetectConfig(**kwargs)
        except Exception:
            # Older versions may not support all kwargs; fall back.
            _FASTTEXT_LID_CONFIG = LangDetectConfig()

    try:
        res = ft_detect(text, model=model_choice, k=1, config=_FASTTEXT_LID_CONFIG)
    except TypeError:
        # Backward-compatible call for older fast-langdetect APIs.
        try:
            res = ft_detect(text)
        except Exception:
            return None
    except Exception:
        return None

    if not res or not isinstance(res, list):
        return None
    cand = res[0]
    if not isinstance(cand, dict):
        return None
    lang = cand.get("lang")
    score = cand.get("score")
    if not isinstance(lang, str):
        return None
    if allowed_langs is not None and lang not in allowed_langs:
        return None
    try:
        score_f = float(score) if score is not None else 0.0
    except Exception:
        score_f = 0.0
    return lang, score_f


def _detect_latin_language(text: str, base_conf: float) -> Tuple[str, float]:
    """Detect which Latin-script language."""
    text_lower = text.lower()

    backend = os.environ.get("LID_BACKEND", "auto").strip().lower()
    if backend in ("fasttext", "auto"):
        ft = _fasttext_langdetect(text_lower, allowed_langs={"en", "de", "es", "fr"})
        if ft is not None:
            lid_lang, lid_score = ft
            conf = base_conf * 0.8 + 0.2 * float(max(0.0, min(1.0, lid_score)))
            return lid_lang, conf

    # Optional lightweight LID backend (langid). This is more robust than
    # regex for short Latin-script outputs and reduces EN/DE/ES/FR confusion.
    if backend in ("langid", "auto"):
        try:
            import langid  # type: ignore

            # Restrict to languages we care about for controls.
            langid.set_languages(["en", "de", "es", "fr"])
            lid_lang, lid_score = langid.classify(text_lower)
            if lid_lang in ("en", "de", "es", "fr"):
                # langid score is uncalibrated; treat it as weak evidence.
                conf = base_conf * 0.8
                return lid_lang, conf
        except Exception:
            pass
    
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


def _detect_arabic_language(text: str, base_conf: float) -> Tuple[str, float]:
    """Disambiguate Arabic-script languages (Urdu vs Arabic).
    
    If LID backend (FastText/LangID) is available, use it to distinguish.
    Otherwise, default to 'ur' but with lower confidence if no Urdu-specific chars found.
    """
    text_lower = text.lower()
    
    # Try FastText (preferred)
    ft = _fasttext_langdetect(text_lower, allowed_langs={"ur", "ar"})
    if ft is not None:
        lid_lang, lid_score = ft
        # Boost confidence if it agrees with script
        final_conf = base_conf * 0.7 + 0.3 * float(max(0.0, min(1.0, lid_score)))
        return lid_lang, final_conf

    # Try LangID
    try:
        import langid
        langid.set_languages(["ur", "ar"])
        lid_lang, _ = langid.classify(text_lower)
        if lid_lang in ("ur", "ar"):
             return lid_lang, base_conf * 0.9
    except ImportError:
        pass

    # Fallback heuristics
    # Urdu specific chars: ٹ ڈ ڑ ں ے ہ
    urdu_chars = set("ٹڈڑںےہ")
    if any(c in urdu_chars for c in text):
        return "ur", base_conf
    
    # Default to 'ur' as per project focus, but lower confidence if ambiguous.
    return "ur", base_conf * 0.8


def is_code_mixed(text: str, threshold: float = 0.2) -> bool:
    """Check if text is code-mixed (multiple scripts present)."""
    ratios = detect_scripts(text)
    significant = [s for s, r in ratios.items() if r > threshold]
    return len(significant) > 1


# =============================================================================
# REPETITION / DEGRADATION
# =============================================================================

def compute_ngram_repetition(text: str, n: int) -> float:
    """Compute n-gram repetition rate.

    By default we measure repetition over whitespace tokens (more robust for
    multilingual text than character n-grams). If the text is too short to
    form token n-grams, we fall back to character n-grams.

    Set REPETITION_UNIT=char to force the legacy character-based metric.
    """
    if not text:
        return 0.0

    unit = os.environ.get("REPETITION_UNIT", "token").strip().lower() or "token"

    def _char_repetition(t: str) -> float:
        if len(t) < n:
            return 0.0
        ngrams = [t[i : i + n] for i in range(len(t) - n + 1)]
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        return (len(ngrams) - len(counts)) / len(ngrams)

    if unit in ("char", "character"):
        return _char_repetition(text)

    tokens = text.split()
    if len(tokens) < n:
        return _char_repetition(text)

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
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
    """CORRECT Jaccard overlap: |A∩B| / |A∪B|, always in [0, 1].
    
    This is the mathematically correct Jaccard similarity coefficient.
    Any value outside [0, 1] indicates a bug in the computation.
    
    Args:
        set_a: First set
        set_b: Second set
        
    Returns:
        Jaccard similarity in range [0, 1]
        
    Raises:
        AssertionError: If result is outside [0, 1] (indicates a bug)
    """
    if not set_a and not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    result = intersection / union
    
    # Research rigor: validate the mathematical invariant
    assert 0.0 <= result <= 1.0, (
        f"BUG: Invalid Jaccard coefficient {result}. "
        f"This should NEVER happen. "
        f"|A|={len(set_a)}, |B|={len(set_b)}, |A∩B|={intersection}, |A∪B|={union}"
    )
    return result


def dice_overlap(set_a: Set, set_b: Set) -> float:
    """Dice coefficient: 2|A∩B| / (|A| + |B|)."""
    if not set_a and not set_b:
        return 0.0
    return 2 * len(set_a & set_b) / (len(set_a) + len(set_b))


###############################################################################
# LLM-AS-JUDGE WITH CALIBRATION
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

    # Fail fast if API key is absent to avoid repeated HTTP errors that would
    # masquerade as missing judge scores in downstream analyses.
    if not api_key:
        return None

    if not api_key or _GEMINI_AVAILABLE is False:
        return None

    _load_llm_judge_cache()
    key = _judge_cache_key(prompt, output, lang_code)
    if key in _LLM_JUDGE_CACHE:
        return _LLM_JUDGE_CACHE[key]

    max_rpm = int(os.environ.get("GEMINI_MAX_RPM", "60"))
    max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        print(f"[eval] LLM judge import/config error: {e}")
        return None

    cfg = LANG_JUDGE_CONFIG.get(lang_code, LANG_JUDGE_CONFIG["hi"])
    prompt_text = JUDGE_PROMPT.format(
        prompt=prompt,
        output=output,
        lang_name=cfg["name"],
        script_name=cfg["script"],
    )

    for attempt in range(max_retries):
        try:
            _respect_gemini_rpm(max_rpm)
            response = model.generate_content(
                prompt_text,
                generation_config={"temperature": 0.0},
            )
            text = response.text.strip()

            # Strip markdown fences
            if text.startswith("```"):
                text = re.sub(r"```\w*\n?", "", text).strip()
                if text.endswith("```"):
                    text = text[:-3].strip()

            parsed = json.loads(text)
            if isinstance(parsed, dict):
                _LLM_JUDGE_CACHE[key] = parsed
                try:
                    _LLM_JUDGE_CACHE_PATH.parent.mkdir(exist_ok=True)
                    with _LLM_JUDGE_CACHE_PATH.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "response": parsed}) + "\n")
                except Exception:
                    pass
                return parsed

            # Unexpected structure; don't retry endlessly.
            return None

        except Exception as e:
            msg = str(e)
            if "429" in msg or "Resource exhausted" in msg:
                backoff = min(60.0, (2 ** attempt) * 2.0 + random.random())
                print(f"[eval] Gemini rate-limited (attempt {attempt+1}/{max_retries}); sleeping {backoff:.1f}s.")
                time.sleep(backoff)
                continue
            print(f"[eval] LLM judge error: {e}")
            return None

    return None


def _clip(x: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clip value to [low, high]."""
    return max(low, min(high, x))


def bias_adjusted_estimator(p: float, q0: float, q1: float) -> float:
    """Compute bias-adjusted accuracy estimate under known judge error rates.

    θ̂ = (p̂ + q₀ - 1) / (q₀ + q₁ - 1)

    Args:
        p: Raw proportion judged correct
        q0: Specificity (judge accuracy on truly incorrect)
        q1: Sensitivity (judge accuracy on truly correct)

    Raises:
        ValueError: If inputs contain NaN or are outside [0, 1]
    """
    # Validate inputs to prevent silent NaN propagation
    import math
    if any(math.isnan(x) for x in [p, q0, q1]):
        raise ValueError(f"NaN input to bias correction: p={p}, q0={q0}, q1={q1}")
    if not all(0.0 <= x <= 1.0 for x in [p, q0, q1]):
        raise ValueError(f"Inputs must be in [0,1]: p={p}, q0={q0}, q1={q1}")

    if q0 + q1 <= 1:
        return p  # Can't correct if judge is worse than random

    theta = (p + q0 - 1) / (q0 + q1 - 1)
    return _clip(theta)


def confidence_interval(
    p: float, q0: float, q1: float,
    n: int, m0: int, m1: int,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute confidence interval for θ under a sensitivity/specificity model.
    
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
    # Guard against division by zero when q0_tilde + q1_tilde ≈ 1
    denom = q0_tilde + q1_tilde - 1
    if abs(denom) < 1e-10:
        return (0.0, 1.0)  # Uninformative CI when correction impossible
    se = sqrt(var_term) / denom
    
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
    
    if n_truly_correct == 0 or n_truly_incorrect == 0:
        print(
            "[eval] Warning: calibration set lacks both correct/incorrect labels; "
            "returning neutral q0=q1=0.5. Calibrated judge estimates will be unreliable."
        )
        return 0.5, 0.5

    q1 = true_positives / n_truly_correct
    q0 = true_negatives / n_truly_incorrect
    
    return q0, q1


def evaluate_with_calibrated_judge(
    test_outputs: List[Dict],  # [{prompt, output, ...}]
    calibration_set: List[Dict],  # [{prompt, output, ground_truth_is_correct}]
    judge_fn=llm_judge_gemini,
    success_threshold: int = 3,  # Language score >= 3 = success
) -> CalibratedJudgeResult:
    """Full evaluation with calibrated LLM-as-judge.
    
    Outline:
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

# Cache calibration tables per path so 2B and 9B runs do not accidentally
# share statistics within the same process.
_JUDGE_CALIBRATION_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}


def load_judge_calibration_table(
    path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Load per-language judge calibration stats from Exp11.

    If `path` is None, we automatically choose a model-matched calibration file:
      - results/exp11_judge_calibration.json for 2B runs
      - results/exp11_judge_calibration_9b.json for 9B runs (USE_9B=1)

    Returns a mapping:
        lang -> {q0, q1, n_calib_0, n_calib_1}

    If the file is missing or malformed, returns an empty dict.
    """
    if path is None:
        use_9b = str(os.environ.get("USE_9B", "0")).lower() in ("1", "true", "yes")
        suffix = "_9b" if use_9b else ""
        path = f"results/exp11_judge_calibration{suffix}.json"

    if path in _JUDGE_CALIBRATION_CACHE:
        return _JUDGE_CALIBRATION_CACHE[path]

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
            if int(n0) < MIN_JUDGE_CALIB_PER_CLASS or int(n1) < MIN_JUDGE_CALIB_PER_CLASS:
                print(
                    f"[eval] Calibration for '{lang}' below MIN_JUDGE_CALIB_PER_CLASS="
                    f"{MIN_JUDGE_CALIB_PER_CLASS} (n0={n0}, n1={n1}); skipping calibrated judge."
                )
                continue
            table[lang] = {
                "q0": float(q0),
                "q1": float(q1),
                "n_calib_0": int(n0),
                "n_calib_1": int(n1),
            }
    except FileNotFoundError:
        print(
            f"[eval] Judge calibration file '{path}' not found; "
            "calibrated summaries will be unavailable."
        )
    except Exception as e:
        print(f"[eval] Error loading judge calibration file '{path}': {e}")

    _JUDGE_CALIBRATION_CACHE[path] = table
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
        print(
            f"[eval] No calibration available for language '{lang}'; "
            "omitting calibrated judge metrics for this language."
        )
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
    semantic_reference: Optional[str] = None,
    qa_references: Any = None,
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
    
    # QA reference handling: allow either a single string or a list of strings.
    qa_refs: List[str] = []
    if qa_references is not None:
        if isinstance(qa_references, list):
            qa_refs = [r for r in qa_references if isinstance(r, str)]
        elif isinstance(qa_references, str):
            qa_refs = [qa_references]
        else:
            qa_refs = [str(qa_references)]

    # Semantic similarity
    sem_sim = -1.0
    sem_ok = False
    ref_text = (
        semantic_reference
        if (semantic_reference is not None and str(semantic_reference).strip())
        else prompt
    )
    
    if compute_semantics and ref_text:
        sem_sim = semantic_similarity(ref_text, output)
        if sem_sim >= 0:
            sem_ok = sem_sim >= SEMANTIC_SIM_THRESHOLD

    # QA metrics (only meaningful when we have a gold reference answer)
    em = f1 = None
    if qa_refs:
        em = qa_exact_match(output, qa_refs)
        f1 = qa_f1(output, qa_refs)
    
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
            def _coerce_score(v: Any) -> float:
                try:
                    x = float(v)
                except Exception:
                    return 0.0
                if not np.isfinite(x):
                    return 0.0
                # Judge rubric is 1–5; clamp defensively.
                return float(max(0.0, min(5.0, x)))

            llm_lang = _coerce_score(candidate.get("language", 0))
            llm_faith = _coerce_score(candidate.get("faithfulness", 0))
            llm_coh = _coerce_score(candidate.get("coherence", 0))
    
    # Overall success: target script + not degraded + semantics OK (if computed)
    success = is_target and not degraded
    if compute_semantics:
        if sem_sim < 0:
            # Semantic model unavailable/failed: treat as failure to avoid optimistic bias.
            success = False
        else:
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
        qa_exact_match=em,
        qa_f1=f1,
        llm_language_score=llm_lang,
        llm_faithfulness_score=llm_faith,
        llm_coherence_score=llm_coh,
        llm_judge_raw=llm_raw,
        overall_success=success,
    )


def _bootstrap_ci_simple(values: List[float], n_bootstrap: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """Simple bootstrap CI computation (avoids circular import with stats.py).

    Default resamples match the paper/config (10,000) for consistency.
    """
    if not values or len(values) < 2:
        mean_val = np.mean(values) if values else 0.0
        return (mean_val, mean_val)

    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    n = int(arr.shape[0])

    # Vectorized bootstrap in chunks (much faster than Python loops and avoids
    # allocating a full [n_bootstrap, n] index matrix at once).
    batch = int(os.environ.get("BOOTSTRAP_BATCH", "1024"))
    batch = max(1, min(batch, n_bootstrap))
    boot_means = np.empty(int(n_bootstrap), dtype=float)
    filled = 0
    while filled < n_bootstrap:
        k = min(batch, n_bootstrap - filled)
        idx = rng.randint(0, n, size=(k, n))
        boot_means[filled : filled + k] = arr[idx].mean(axis=1)
        filled += k

    return (
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def estimate_power_binary(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    """Approximate power for difference in proportions (normal approx).

    Args:
        p1, p2: proportions to compare (baseline vs treatment)
        n: number of paired samples (treated as per-condition count)
        alpha: significance level

    Returns:
        Approximate power (0-1)
    """
    import math
    if n <= 0:
        return 0.0
    p_bar = 0.5 * (p1 + p2)
    se = math.sqrt(2 * p_bar * (1 - p_bar) / n)
    if se == 0:
        return 0.0
    z = (p2 - p1) / se
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    # two-sided approximation
    power = norm.cdf(z - z_alpha) + (1 - norm.cdf(z + z_alpha))
    return float(max(0.0, min(1.0, power)))


def estimate_power_paired(values_a: List[float], values_b: List[float], alpha: float = 0.05) -> Optional[float]:
    """Approximate power for a paired mean-difference test using normal approx.

    This is a lightweight approximation based on paired Cohen's d:
      d = mean(diff) / std(diff)
      z ≈ |d| * sqrt(n)
      power ≈ Φ(z - z_alpha)
    """
    if not values_a or not values_b:
        return None
    if len(values_a) != len(values_b):
        return None
    if len(values_a) < 2:
        return None
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = b - a
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return None
    d = float(np.mean(diff) / sd)
    z_alpha = norm.ppf(1 - alpha / 2)
    z = abs(d) * float(np.sqrt(len(diff)))
    power = norm.cdf(z - z_alpha)
    return float(max(0.0, min(1.0, power)))


def estimate_power_independent(values_a: List[float], values_b: List[float], alpha: float = 0.05) -> Optional[float]:
    """Approximate power for an independent two-sample mean-difference test.

    Uses a large-sample normal approximation with Cohen's d and
    n_eff = n1*n2/(n1+n2).
    """
    if not values_a or not values_b:
        return None
    if len(values_a) < 2 or len(values_b) < 2:
        return None

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n1 = int(a.shape[0])
    n2 = int(b.shape[0])
    if n1 < 2 or n2 < 2:
        return None

    var1 = float(np.var(a, ddof=1))
    var2 = float(np.var(b, ddof=1))
    denom = (n1 + n2 - 2)
    if denom <= 0:
        return None
    pooled = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / denom))
    if pooled == 0.0 or not np.isfinite(pooled):
        return None

    d = float((np.mean(b) - np.mean(a)) / pooled)
    n_eff = float(n1 * n2 / (n1 + n2))
    z = abs(d) * float(np.sqrt(n_eff))
    z_alpha = norm.ppf(1 - alpha / 2)
    power = norm.cdf(-z_alpha - z) + (1 - norm.cdf(z_alpha - z))
    return float(max(0.0, min(1.0, power)))


def aggregate_results(
    results: List[SteeringEvalResult],
    target_script: str = "devanagari",
    _warn_on_default: bool = True,
    compute_separate_metrics: bool = True,
    sensitivity_delta: float = 0.05,
) -> AggregateResults:
    """Aggregate evaluation results with separate metrics and bootstrap CIs.

    Args:
        results: List of SteeringEvalResult objects.
        target_script: Script name to use when computing
            avg_target_script_ratio. This should match the target language
            for the steering experiment, e.g. "devanagari" for Hindi,
            "bengali" for Bengali, "tamil" for Tamil, etc.
        _warn_on_default: Internal flag; set False to suppress the default warning.
        compute_separate_metrics: If True, compute detailed SeparateMetrics with CIs.

    Note:
        Always explicitly pass target_script for non-Hindi languages to avoid
        incorrect script ratio calculations.
    """
    # Research rigor: warn if using default script for potentially non-Hindi evaluation
    if _warn_on_default and target_script == "devanagari" and results:
        # Check if any result has a detected language that isn't Hindi
        non_hindi_detected = any(
            r.detected_language not in ("hi", "unknown", "en")
            for r in results
            if r.detected_language
        )
        if non_hindi_detected:
            import warnings
            warnings.warn(
                "aggregate_results() using default target_script='devanagari' but "
                "some outputs were detected as non-Hindi. Consider passing explicit "
                "target_script for accurate script ratio calculation.",
                UserWarning,
            )

    if not results:
        return AggregateResults(
            n_samples=0, success_rate=0.0, semantic_success_rate=None,
            avg_target_script_ratio=0.0, avg_semantic_similarity=None,
            avg_repetition_3gram=0.0, avg_repetition_5gram=0.0,
            degradation_rate=0.0, n_empty_outputs=0
        )

    n = len(results)

    # Per-prompt binary indicators (for downstream statistical testing)
    per_prompt_script = [1 if r.is_target_script else 0 for r in results]
    per_prompt_degraded = [1 if r.is_degraded else 0 for r in results]

    # Script success
    script_success = sum(per_prompt_script) / n

    # Target script ratio and empty-output tracking (no alphabetic chars)
    n_empty_outputs = sum(1 for r in results if not r.script_ratios)
    target_ratios = [r.script_ratios.get(target_script, 0.0) for r in results]

    # Semantic
    sem_values = [r.semantic_similarity for r in results if r.semantic_similarity >= 0]
    per_prompt_semantic = [r.semantic_similarity for r in results]  # May include -1.0
    avg_sem = float(np.mean(sem_values)) if sem_values else None

    # Semantic + script success
    sem_success = None
    if sem_values:
        sem_success_count = sum(1 for r in results if r.overall_success)
        sem_success = sem_success_count / n

    # Sensitivity analysis: script ratio and semantic thresholds ± delta
    sensitivity = None
    if compute_separate_metrics and n >= 5:
        thr_script = DEVANAGARI_THRESHOLD
        thr_sem = SEMANTIC_SIM_THRESHOLD
        variants = []
        for ds in (-sensitivity_delta, 0.0, sensitivity_delta):
            t_script = max(0.0, min(1.0, thr_script + ds))
            t_sem = thr_sem + ds
            variant_success = 0
            for r in results:
                is_script = r.script_ratios.get(target_script, 0) >= t_script
                is_sem = (r.semantic_similarity >= t_sem) if r.semantic_similarity >= 0 else False
                success_v = is_script and (not r.is_degraded) and (not sem_values or is_sem)
                variant_success += 1 if success_v else 0
            variants.append({
                "delta": ds,
                "threshold_script": t_script,
                "threshold_sem": t_sem,
                "success_rate": variant_success / n,
            })
        sensitivity = variants

    # Degradation
    deg_rate = sum(per_prompt_degraded) / n

    # Compute separate metrics with bootstrap CIs if requested
    separate_metrics = None
    if compute_separate_metrics and n >= 5:  # Need minimum samples for meaningful CI
        # Script success CI
        script_ci = _bootstrap_ci_simple(per_prompt_script)

        # Script ratio CI
        ratio_ci = _bootstrap_ci_simple(target_ratios)

        # Semantic CI (only for valid values)
        sem_ci = None
        sem_pres_rate = None
        if sem_values:
            sem_ci = _bootstrap_ci_simple(sem_values)
            sem_pres_rate = sum(1 for v in sem_values if v >= SEMANTIC_SIM_THRESHOLD) / len(sem_values)

        # Degradation CI
        deg_ci = _bootstrap_ci_simple(per_prompt_degraded)

        # Combined success (script AND semantic AND not-degraded)
        combined_success = sum(1 for r in results if r.overall_success) / n

        separate_metrics = SeparateMetrics(
            script_success_rate=script_success,
            script_success_ci=script_ci,
            avg_script_ratio=float(np.mean(target_ratios)),
            script_ratio_ci=ratio_ci,
            semantic_mean=avg_sem,
            semantic_ci=sem_ci,
            semantic_preservation_rate=sem_pres_rate,
            degradation_rate=deg_rate,
            degradation_ci=deg_ci,
            avg_repetition_3gram=float(np.mean([r.repetition_3gram for r in results])),
            avg_repetition_5gram=float(np.mean([r.repetition_5gram for r in results])),
            combined_success_rate=combined_success,
        )

    return AggregateResults(
        n_samples=n,
        success_rate=script_success,
        semantic_success_rate=sem_success,
        avg_target_script_ratio=float(np.mean(target_ratios)),
        avg_semantic_similarity=avg_sem,
        avg_repetition_3gram=float(np.mean([r.repetition_3gram for r in results])),
        avg_repetition_5gram=float(np.mean([r.repetition_5gram for r in results])),
        degradation_rate=deg_rate,
        n_empty_outputs=n_empty_outputs,
        separate_metrics=separate_metrics,
        per_prompt_script_success=per_prompt_script,
        per_prompt_semantic_sim=per_prompt_semantic,
        per_prompt_degraded=per_prompt_degraded,
        sensitivity=sensitivity,
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
    print("3. LLM-as-Judge bias correction...")
    # Example: q0=0.7, q1=0.9 (imperfect judge)
    p_raw = 0.5
    theta = bias_adjusted_estimator(p_raw, q0=0.7, q1=0.9)
    print(f"   Raw: {p_raw:.2f}, Corrected: {theta:.2f}")

    ci = confidence_interval(p_raw, 0.7, 0.9, n=100, m0=50, m1=50)
    print(f"   95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print("   ✓ Bias correction implemented\n")

    # 4. Language detection (script-based + LID)
    print("4. Language detection tests...")

    # 4a. Devanagari should detect as Hindi
    hindi_text = "नमस्ते, मेरा नाम राहुल है।"
    hi_lang, hi_conf = detect_language(hindi_text)
    assert hi_lang == "hi", f"Hindi text detected as '{hi_lang}', expected 'hi'"
    print(f"   Hindi: '{hi_lang}' (conf={hi_conf:.2f}) ✓")

    # 4b. Bengali script
    bengali_text = "আমি বাংলায় কথা বলি।"
    bn_lang, bn_conf = detect_language(bengali_text)
    assert bn_lang == "bn", f"Bengali text detected as '{bn_lang}', expected 'bn'"
    print(f"   Bengali: '{bn_lang}' (conf={bn_conf:.2f}) ✓")

    # 4c. Latin-script English vs German (requires LID backend)
    english_text = "The quick brown fox jumps over the lazy dog."
    en_lang, en_conf = detect_language(english_text)
    print(f"   English: '{en_lang}' (conf={en_conf:.2f})")

    german_text = "Der schnelle braune Fuchs springt über den faulen Hund."
    de_lang, de_conf = detect_language(german_text)
    print(f"   German: '{de_lang}' (conf={de_conf:.2f})")

    # 4d. Arabic script: Urdu vs Arabic (the hard case!)
    # Note: These tests verify the detection runs; actual accuracy depends on LID backend
    urdu_text = "میں پاکستان سے ہوں اور اردو بولتا ہوں۔"  # "I am from Pakistan and speak Urdu"
    ur_lang, ur_conf = detect_language(urdu_text)
    print(f"   Urdu text: '{ur_lang}' (conf={ur_conf:.2f})")

    arabic_text = "أنا من مصر وأتكلم العربية."  # "I am from Egypt and speak Arabic"
    ar_lang, ar_conf = detect_language(arabic_text)
    print(f"   Arabic text: '{ar_lang}' (conf={ar_conf:.2f})")

    # Check that at least the script is detected correctly (Arabic script for both)
    ur_ratios = detect_scripts(urdu_text)
    ar_ratios = detect_scripts(arabic_text)
    assert ur_ratios.get("arabic", 0) > 0.5, "Urdu should have Arabic script"
    assert ar_ratios.get("arabic", 0) > 0.5, "Arabic should have Arabic script"
    print("   ✓ Arabic script detection works for both UR and AR")

    # 4e. Confidence-aware API
    print("\n5. Confidence-aware language detection...")
    lang, conf, reliable = get_language_with_confidence(hindi_text)
    print(f"   Hindi: lang='{lang}', conf={conf:.2f}, reliable={reliable}")
    assert reliable, "Hindi with clear Devanagari should be reliable"

    lang, conf, reliable = get_language_with_confidence("xyz")  # Ambiguous short text
    print(f"   Short text: lang='{lang}', conf={conf:.2f}, reliable={reliable}")
    print("   ✓ Confidence-aware API works\n")

    print("✓ All tests passed!")
