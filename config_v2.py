"""Research-Grade Configuration for SAE Multilingual Steering Experiments.

This configuration is designed for publication-quality experiments with:
- Literature-backed sample sizes and metrics
- Train/test split to prevent data leakage
- Coherence and capability preservation evaluation
- Support for multiple model sizes (2B, 9B)

Based on:
- O'Brien et al. 2024: "Steering Language Model Refusal with SAEs" (capability degradation)
- Gao et al. 2024: "Scaling and Evaluating Sparse Autoencoders" (SAE metrics)
- Vogels et al. 2025: "In-Distribution Steering" (coherence preservation)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")  # Free from aistudio.google.com

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
@dataclass
class ModelConfig:
    """Configuration for model and SAE."""
    name: str
    hf_id: str
    sae_release: str
    n_layers: int
    hidden_dim: int
    sae_width: int
    
# Gemma 2 2B (primary - faster iteration)
GEMMA_2B = ModelConfig(
    name="gemma-2-2b",
    hf_id="google/gemma-2-2b",
    sae_release="gemma-scope-2b-pt-res-canonical",
    n_layers=26,
    hidden_dim=2304,
    sae_width=16384,  # 16k width
)

# Gemma 2 9B (secondary - validation)
GEMMA_9B = ModelConfig(
    name="gemma-2-9b",
    hf_id="google/gemma-2-9b",
    sae_release="gemma-scope-9b-pt-res-canonical",
    n_layers=42,
    hidden_dim=3584,
    sae_width=131072,  # 131k width
)

# Active model (change this to switch)
ACTIVE_MODEL = GEMMA_2B

# ============================================================================
# RESEARCH-GRADE SAMPLE SIZES
# ============================================================================
"""
Sample size justification (literature-backed):

1. Feature Discovery:
   - Gao et al. 2024 uses ~1000-10000 samples for SAE analysis
   - We use 5000 for discovery (train) and 1000 for validation (test)
   
2. Steering Evaluation:
   - O'Brien et al. 2024 evaluates on multiple benchmarks
   - We use 200 diverse prompts per condition
   
3. Statistical Power:
   - 3 random seeds minimum for confidence intervals
   - Bootstrap with 1000 resamples for p-values

Reference sample sizes from literature:
- Gemma Scope (Lieberum 2024): ~1000 samples per analysis
- Do Llamas Work in English (Wendler 2024): ~500 parallel sentences
- SAE-SSV (He 2025): 1000+ samples for sentiment/truthfulness steering
"""

# Sample sizes
N_SAMPLES_DISCOVERY_TRAIN = 5000   # For feature discovery (train split)
N_SAMPLES_DISCOVERY_TEST = 1000    # For feature validation (test split)
N_SAMPLES_STEERING_EVAL = 200      # Diverse prompts for steering evaluation
N_RANDOM_SEEDS = 3                 # Number of runs for confidence intervals

# Data split ratio
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================
# Primary languages
LANGUAGES = ["en", "hi", "ur"]  # English, Hindi, Urdu

# Extended Indic languages (for generalization testing)
INDIC_LANGUAGES = ["hi", "bn", "ta", "te", "mr", "gu", "pa", "ml"]  # Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Malayalam

# Non-Indic control languages
NON_INDIC_LANGUAGES = ["de", "fr", "zh", "ja", "ar", "ru"]  # German, French, Chinese, Japanese, Arabic, Russian

# Script families (for semantic vs script analysis)
SCRIPT_FAMILIES = {
    "devanagari": ["hi", "mr", "ne"],      # Hindi, Marathi, Nepali
    "arabic": ["ur", "ar", "fa"],          # Urdu, Arabic, Persian
    "latin": ["en", "de", "fr", "es"],     # English, German, French, Spanish
    "dravidian": ["ta", "te", "ml", "kn"], # Tamil, Telugu, Malayalam, Kannada
}

# Target language for steering
TARGET_LANGUAGE = "hi"  # Hindi

# ============================================================================
# LAYER CONFIGURATION
# ============================================================================
"""
Layer selection strategy:

Based on "Do Llamas Work in English?" (Wendler 2024):
- Early layers (0-30%): Token embedding, basic processing
- Middle layers (30-60%): Concept space (language-agnostic)
- Late layers (60-100%): Output preparation (language-specific)

For Gemma 2 2B (26 layers):
- Early: 0-7 (0-27%)
- Middle: 8-15 (31-58%)
- Late: 16-25 (62-96%)

Your results show PEAK at layer 24 (late), MINIMUM at 13-16 (middle).
This contradicts naive "Messy Middle" but supports concept→output transition.
"""

# Gemma 2 2B layer ranges
LAYERS_2B = {
    "early": [3, 5, 7],
    "middle": [10, 13, 15],  # "Messy Middle" - where H3 predicts peak
    "late": [18, 21, 24],
}
TARGET_LAYERS_2B = [5, 8, 10, 13, 16, 20, 24]  # Full sweep

# Gemma 2 9B layer ranges (42 layers)
LAYERS_9B = {
    "early": [5, 8, 12],
    "middle": [18, 21, 24],
    "late": [30, 36, 40],
}
TARGET_LAYERS_9B = [8, 14, 21, 28, 35, 40]

# Active layers
TARGET_LAYERS = TARGET_LAYERS_2B if ACTIVE_MODEL == GEMMA_2B else TARGET_LAYERS_9B

# ============================================================================
# FEATURE SELECTION THRESHOLDS
# ============================================================================
"""
Your verification results show:
- Monolinguality features: M = 16,017,084 (max), but 0% steering success
- Activation-diff features: Hindi/English ratio = 1.8x, but 100% success
- Overlap between top-25: 0%

Key insight: High monolinguality ≠ good for steering
Monolinguality selects "detectors" (presence features)
Activation-diff selects "generators" (transformation features)
"""

# Monolinguality threshold (for analysis, not steering)
MONOLINGUALITY_THRESHOLD = 3.0  # M > 3.0 means 3x more active for target language

# Activation threshold for "active" features
ACTIVATION_THRESHOLD = 0.1  # Features with max activation > 0.1 are "active"

# Number of features to select for steering
N_STEERING_FEATURES = 25  # Top-k features for steering

# Steering strengths to test
STEERING_STRENGTHS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]

# ============================================================================
# EVALUATION METRICS
# ============================================================================
"""
Based on O'Brien et al. 2024 and Gao et al. 2024:

1. Steering Success Metrics:
   - Script detection (Devanagari Unicode ratio)
   - LLM-as-judge (Gemini)
   
2. Capability Preservation Metrics:
   - Perplexity (fluency)
   - Repetition detection (coherence)
   - Task performance (general capability)
   
3. Feature Quality Metrics (from Gao 2024):
   - Reconstruction loss
   - L0 sparsity
   - Downstream effect sparsity
"""

# Success threshold for script detection
SCRIPT_DETECTION_THRESHOLD = 0.3  # At least 30% Devanagari characters

# Repetition thresholds (degradation indicators)
REPETITION_3GRAM_THRESHOLD = 0.3   # Warning level
REPETITION_5GRAM_THRESHOLD = 0.2   # Critical level

# Perplexity increase threshold (capability degradation)
PERPLEXITY_INCREASE_THRESHOLD = 2.0  # Max 2x increase from baseline

# ============================================================================
# LLM-AS-JUDGE CONFIGURATION
# ============================================================================
LLM_JUDGE_PROVIDER = "gemini"  # Options: "gemini", "claude", "openai"
LLM_JUDGE_MODEL = "gemini-2.0-flash"  # FREE! Use "gemini-1.5-pro" for higher quality

# Evaluation prompt template
LLM_JUDGE_PROMPT = """You are evaluating whether a language model successfully generated Hindi text.

Task: Determine if the output is PRIMARILY in Hindi (Devanagari script).

Evaluation criteria:
1. Is the majority of the text in Hindi (Devanagari script)?
2. Is the text coherent and meaningful?
3. Does the text respond appropriately to the prompt?

Prompt: {prompt}
Output: {output}

Respond ONLY with a JSON object (no markdown, no explanation):
{{"is_hindi": true/false, "coherence_score": 1-5, "relevance_score": 1-5, "reasoning": "brief explanation"}}"""

# ============================================================================
# CAPABILITY PRESERVATION BENCHMARKS
# ============================================================================
"""
Based on O'Brien et al. 2024:
"SAE steering shows systematic degradation of performance across multiple 
benchmark tasks, even on safe inputs with no apparent connection to refusal."

We measure:
1. General knowledge (TriviaQA-style)
2. Reasoning (simple math/logic)
3. Instruction following (basic tasks)
"""

CAPABILITY_BENCHMARK_PROMPTS = [
    # General knowledge
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
    "How many continents are there?",
    "What planet is known as the Red Planet?",
    
    # Simple reasoning
    "If I have 3 apples and give away 1, how many do I have?",
    "What comes next in the sequence: 2, 4, 6, 8, ?",
    "If all dogs are animals, and Buddy is a dog, is Buddy an animal?",
    "What is 15 + 27?",
    "If today is Monday, what day is tomorrow?",
    
    # Instruction following
    "List three colors.",
    "Write a word that rhymes with 'cat'.",
    "Name one country in Europe.",
    "What is the opposite of 'hot'?",
    "Count from 1 to 5.",
]

# ============================================================================
# SEMANTIC VS SYNTACTIC DISTINCTION
# ============================================================================
"""
Key distinction (for your thesis):

SEMANTIC features encode MEANING:
- Activated by concepts regardless of language
- Example: "dog" (en), "कुत्ता" (hi), "کتا" (ur) → same semantic feature

SYNTACTIC/SCRIPT features encode FORM:
- Activated by specific writing systems or grammatical structures
- Example: Devanagari characters → Hindi script feature
- Example: SOV word order → Hindi syntax feature

Your H4 results show 94-99% Hindi-Urdu overlap because they share SEMANTICS
(same spoken language) but differ in SCRIPT (Devanagari vs Arabic).

To disentangle:
1. Semantic features: High overlap across Hindi-Urdu, moderate with English
2. Script features: NO overlap between Hindi (Devanagari) and Urdu (Arabic)
3. Syntactic features: High overlap Hindi-Urdu (both SOV), low with English (SVO)
"""

# Script detection patterns (Unicode ranges)
SCRIPT_UNICODE_RANGES = {
    "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit
    "arabic": (0x0600, 0x06FF),       # Urdu, Arabic
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "latin": (0x0041, 0x007A),        # A-Z, a-z
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
CHECKPOINTS_DIR = "checkpoints"

# ============================================================================
# COMPUTATIONAL BUDGET
# ============================================================================
BATCH_SIZE = 8
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
USE_FLASH_ATTENTION = True  # Requires flash-attn package

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEEDS = [42, 123, 456]  # For 3-run experiments
DEFAULT_SEED = 42
