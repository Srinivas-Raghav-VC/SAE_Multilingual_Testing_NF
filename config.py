"""Configuration for SAE Multilingual Steering Research.

This config supports:
- Multiple datasets (Samanantar, FLORES, MLQA, IndicQA)
- Language clustering analysis
- Proper train/test/eval splits
"""

import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default base model for most experiments (Gemma 2 2B)
MODEL_ID_2B = "google/gemma-2-2b"
MODEL_ID = MODEL_ID_2B
MODEL_NAME = MODEL_ID  # Alias
N_LAYERS_2B = 26
HIDDEN_DIM_2B = 2304

# Optional: larger model for scaling experiments
MODEL_ID_9B = "google/gemma-2-9b"
N_LAYERS_9B = 42
HIDDEN_DIM_9B = 3584  # Gemma 2 9B hidden size (official config)

# Global scaling toggle used across repo.
USE_9B = str(os.environ.get("USE_9B", "0")).lower() in ("1", "true", "yes")
if USE_9B:
    MODEL_ID = MODEL_ID_9B
    MODEL_NAME = MODEL_ID
    N_LAYERS = N_LAYERS_9B
    HIDDEN_DIM = HIDDEN_DIM_9B
else:
    N_LAYERS = N_LAYERS_2B
    HIDDEN_DIM = HIDDEN_DIM_2B

# SAE Configuration (Gemma Scope releases)
# These match the official Hugging Face repos:
#   - google/gemma-scope-2b-pt-res
#   - google/gemma-scope-9b-pt-res
SAE_RELEASE_2B = "gemma-scope-2b-pt-res-canonical"
SAE_RELEASE_9B = "gemma-scope-9b-pt-res-canonical"
SAE_RELEASE = SAE_RELEASE_9B if USE_9B else SAE_RELEASE_2B

# SAE width configuration
# Available widths in Gemma Scope: "16k", "32k", "65k", "131k", "262k", "524k", "1m"
# Default to 16k for balance between interpretability and coverage
SAE_WIDTH_2B = "16k"
SAE_WIDTH_9B = "16k"
SAE_WIDTH = SAE_WIDTH_9B if USE_9B else SAE_WIDTH_2B

# Attention implementation
# Options: "eager" (default, compatible), "flash_attention_2" (faster on A100/H100)
# Set USE_FLASH_ATTN=1 environment variable to enable flash attention on supported GPUs
USE_FLASH_ATTN_ENV = str(os.environ.get("USE_FLASH_ATTN", "0")).lower() in ("1", "true", "yes")
ATTN_IMPLEMENTATION = "flash_attention_2" if USE_FLASH_ATTN_ENV else "eager"

# =============================================================================
# TARGET LAYERS
# =============================================================================
# 2B sweep (26 layers): absolute probes used throughout the original runs.
TARGET_LAYERS_2B = [5, 8, 10, 13, 16, 20, 24]
# 9B sweep (42 layers): relative-depth matched probes (see archive/config_v2.py).
TARGET_LAYERS_9B = [8, 14, 21, 28, 35, 40]

TARGET_LAYERS = TARGET_LAYERS_9B if USE_9B else TARGET_LAYERS_2B

# Hard validation: layer indices must be valid for the selected model.
for _layer in TARGET_LAYERS:
    if not isinstance(_layer, int):
        raise TypeError(f"TARGET_LAYERS must contain ints, got {type(_layer).__name__}: {_layer!r}")
    if _layer < 0 or _layer >= N_LAYERS:
        raise ValueError(
            f"TARGET_LAYERS contains out-of-range layer {_layer} for MODEL_ID={MODEL_ID} "
            f"(valid: 0–{N_LAYERS-1})."
        )

LAYER_RANGES_2B = {
    "early": [3, 5, 7],
    "middle": [10, 13, 15],
    "late": [18, 21, 24],
}
LAYER_RANGES_9B = {
    "early": [5, 8, 12],
    "middle": [18, 21, 24],
    "late": [30, 36, 40],
}

LAYER_RANGES = LAYER_RANGES_9B if USE_9B else LAYER_RANGES_2B

for _group, _layers in LAYER_RANGES.items():
    for _layer in _layers:
        if _layer < 0 or _layer >= N_LAYERS:
            raise ValueError(
                f"LAYER_RANGES['{_group}'] contains out-of-range layer {_layer} for MODEL_ID={MODEL_ID} "
                f"(valid: 0–{N_LAYERS-1})."
            )

# =============================================================================
# LANGUAGES
# =============================================================================

# FLORES-200 codes - Core languages for experiments
LANGUAGES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",  # Kannada (Dravidian) - ADDED for complete Dravidian coverage
    "ml": "mal_Mlym",  # Malayalam (Dravidian) - ADDED for complete Dravidian coverage
    "de": "deu_Latn",
    "ar": "arb_Arab",
}

# Extended languages for clustering analysis
EXTENDED_LANGUAGES = {
    # Indic - Indo-Aryan
    "hi": "hin_Deva",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    
    # Indic - Dravidian
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    
    # Control languages
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    # Additional low-resource / control
    "vi": "vie_Latn",
}

TARGET_LANGUAGE = "hi"

# =============================================================================
# LANGUAGE FAMILY GROUPINGS (for Indic cluster analysis)
# =============================================================================

# Indic languages grouped by family
INDIC_LANGUAGES = {
    "indo_aryan": ["hi", "ur", "bn", "mr", "gu", "pa", "or", "as"],
    "dravidian": ["ta", "te", "kn", "ml"],
}

# All Indic language codes (flat list)
ALL_INDIC = INDIC_LANGUAGES["indo_aryan"] + INDIC_LANGUAGES["dravidian"]

# Control (non-Indic) languages
CONTROL_LANGUAGES = {
    "germanic": ["en", "de"],
    "semitic": ["ar"],
    "sino_tibetan": ["zh"],
}

# =============================================================================
# TYPOLOGICAL FEATURES (for areal vs genetic clustering analysis)
# =============================================================================
# These features help distinguish whether clustering is due to:
# - Genetic family (Indo-Aryan vs Dravidian)
# - Areal features (shared due to contact, e.g., retroflexes)
# - Typological properties (SOV word order, agglutination)

TYPOLOGICAL_FEATURES = {
    # Retroflex consonants: Shared Indo-Aryan + Dravidian areal feature
    # (borrowed into Indo-Aryan from Dravidian substrate)
    "retroflex": ["hi", "ur", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "as"],
    "no_retroflex": ["en", "de", "ar", "zh", "fr", "es"],

    # Word order
    "sov": ["hi", "ur", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "as"],  # All Indic are SOV
    "svo": ["en", "zh", "vi"],
    "vso": ["ar"],

    # Agglutination scale (1=analytic, 5=highly agglutinative)
    # Dravidian languages are more agglutinative than Indo-Aryan
    "agglutination": {
        # Dravidian (higher)
        "ta": 5, "kn": 4, "ml": 4, "te": 3,
        # Indo-Aryan (moderate)
        "hi": 2, "ur": 2, "bn": 2, "mr": 2, "gu": 2, "pa": 2, "or": 2, "as": 2,
        # Analytic
        "en": 1, "zh": 1,
        # Semitic (templatic, not agglutinative)
        "ar": 2,
    },
}

# Typological control languages (for SOV without Indic genetic relation)
TYPOLOGICAL_CONTROLS = {
    "ja": "jpn_Jpan",  # Japanese: SOV, agglutinative, no Indic relation
    "ko": "kor_Hang",  # Korean: SOV, agglutinative, no Indic relation
    "tr": "tur_Latn",  # Turkish: SOV, agglutinative, no Indic relation
}

# =============================================================================
# SAMPLE SIZES
# =============================================================================

# For feature discovery (use Samanantar - 49M sentences!)
N_SAMPLES_DISCOVERY = 5000
N_SAMPLES_DISCOVERY_TRAIN = 4000
N_SAMPLES_DISCOVERY_TEST = 1000

# For steering evaluation (use MLQA/IndicQA)
N_SAMPLES_EVAL = 200
N_STEERING_EVAL = 200

# Train/test split
TRAIN_TEST_SPLIT = 0.8

# =============================================================================
# MINIMUM SAMPLE SIZES FOR RESEARCH RIGOR
# =============================================================================
# These are minimum thresholds for statistically meaningful results.
# Experiments will warn if sample sizes fall below these thresholds.

MIN_SAMPLES_PER_LANGUAGE = 100      # Minimum texts per language for feature discovery
MIN_PROMPTS_STEERING = 30           # Minimum prompts for steering evaluation
MIN_PROMPTS_CAUSAL_PROBING = 20     # Minimum prompts for causal effect estimation
MIN_FEATURES_FOR_STEERING = 5       # Minimum features to consider for steering vector

# =============================================================================
# THRESHOLDS
# =============================================================================

# Monolinguality: M > 3.0 means feature 3x more active for target language
MONOLINGUALITY_THRESHOLD = 3.0

# Activation threshold for "active" feature
ACTIVATION_THRESHOLD = 0.1

# Script detection (success if >= threshold in target script)
SCRIPT_DETECTION_THRESHOLD = 0.3
DEVANAGARI_THRESHOLD = 0.3

# Repetition thresholds (degradation indicators)
REPETITION_3GRAM_THRESHOLD = 0.3
REPETITION_5GRAM_THRESHOLD = 0.2

# Semantic evaluation (LaBSE + threshold)
# LaBSE is multilingual and strong on Indic languages, making it suitable
# for semantic-preservation checks in steering experiments.
SEMANTIC_MODEL_NAME = "sentence-transformers/LaBSE"
# Cosine similarity threshold for “semantics preserved”.
# 0.7 is a standard strong-similarity cutoff in multilingual ST literature.
SEMANTIC_SIM_THRESHOLD = 0.7

# =============================================================================
# STEERING CONFIGURATION
# =============================================================================

STEERING_STRENGTHS = [0.5, 1.0, 2.0, 4.0, 8.0]
NUM_FEATURES = 25
NUM_FEATURES_OPTIONS = [10, 25, 50, 100]  # For sensitivity analysis

STEERING_METHODS = ["activation_diff", "monolinguality", "random", "dense"]

# =============================================================================
# SENSITIVITY ANALYSIS THRESHOLDS
# =============================================================================
# These are used for robustness checks - experiments should report results
# at multiple threshold values to show stability.

SENSITIVITY_THRESHOLDS = {
    # Jaccard overlap: feature activation rate thresholds
    "jaccard_activation_rates": [0.01, 0.02, 0.05, 0.1],

    # Monolinguality: different selectivity thresholds
    "monolinguality_thresholds": [2.0, 3.0, 5.0, 10.0],

    # Script detection: success thresholds
    "script_success_thresholds": [0.3, 0.5, 0.7, 0.85],

    # Semantic similarity: preservation thresholds
    "semantic_thresholds": [0.5, 0.6, 0.7, 0.8],

    # Feature counts for steering
    "num_features_options": [10, 25, 50, 100],
}

# =============================================================================
# STATISTICAL TESTING CONFIGURATION
# =============================================================================

STATISTICAL_CONFIG = {
    # Bootstrap
    "n_bootstrap": 10000,
    "confidence_level": 0.95,
    "seed": 42,

    # Multiple comparison correction
    "correction_method": "holm",  # "bonferroni", "holm", "fdr"
    "alpha": 0.05,

    # Effect size interpretation thresholds (Cohen's d)
    "effect_size_small": 0.2,
    "effect_size_medium": 0.5,
    "effect_size_large": 0.8,

    # Minimum samples for reliable statistics
    "min_samples_for_ci": 10,
    "min_samples_for_test": 5,
}

# =============================================================================
# EVALUATION PROMPTS
# =============================================================================
# These prompts are used for steering evaluation across experiments.
# For statistical reliability, we recommend at least MIN_PROMPTS_STEERING prompts.
# Additional prompts are loaded from FLORES English sentences at runtime.

EVAL_PROMPTS = [
    # Simple questions
    "Hello, how are you today?",
    "What is your name?",
    "What is the weather like?",
    "What time is it?",
    "Where do you live?",
    
    # Continuations
    "Once upon a time, there was a",
    "The story begins with",
    "In a small village,",
    "Long ago, in a distant land,",
    "The sun rose over the",
    
    # India-related (should encourage Hindi)
    "The capital of India is",
    "Tell me about Indian culture.",
    "Indian food is known for",
    "The Taj Mahal is",
    "Bollywood movies are",
    
    # General knowledge
    "My favorite food is",
    "I want to learn about",
    "The best way to travel is",
    "Science tells us that",
    "The future of technology is",
    
    # Instructions
    "Write a poem about nature.",
    "Describe a beautiful sunset.",
    "Tell me a story about friendship.",
    "Explain how to make tea.",
    "List three important things in life.",
]

# =============================================================================
# LANGUAGE TO SCRIPT MAPPING
# =============================================================================
# Centralized mapping from language codes to script names used in evaluation.
# This ensures consistent target_script handling across all experiments.

LANG_TO_SCRIPT = {
    "hi": "devanagari",
    "ur": "arabic",
    "bn": "bengali",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "gu": "gujarati",
    "pa": "gurmukhi",
    "or": "oriya",
    "mr": "devanagari",  # Marathi uses Devanagari
    "as": "bengali",     # Assamese uses Bengali script variant
    "de": "latin",
    "en": "latin",
    "fr": "latin",
    "es": "latin",
    "ar": "arabic",
    "zh": "han",
}

# =============================================================================
# UNICODE RANGES FOR SCRIPT DETECTION
# =============================================================================

# Script ranges now support multiple ranges per script for complete coverage
# Format: script_name -> list of (start, end) tuples
SCRIPT_RANGES = {
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],  # + Extended Devanagari
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF)],  # + Supplement + Forms-A
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "oriya": [(0x0B00, 0x0B7F)],
    # Latin: Basic + Latin-1 Supplement + Extended-A + Extended-B
    # This covers German umlauts (ä, ö, ü), French accents, etc.
    "latin": [(0x0041, 0x007A), (0x00C0, 0x00FF), (0x0100, 0x017F), (0x0180, 0x024F)],
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],  # + Extension A
}

# Legacy single-range format for backward compatibility
SCRIPT_RANGES_SINGLE = {
    "devanagari": (0x0900, 0x097F),
    "arabic": (0x0600, 0x06FF),
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "gujarati": (0x0A80, 0x0AFF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "oriya": (0x0B00, 0x0B7F),
    "latin": (0x0041, 0x007A),
    "han": (0x4E00, 0x9FFF),
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
CHECKPOINTS_DIR = "checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# =============================================================================
# COMPUTATIONAL
# =============================================================================

BATCH_SIZE = 8
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
USE_FLASH_ATTENTION = True
SEED = 42

# Environment
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Support both GOOGLE_API_KEY and GEMINI_API_KEY environment variable names.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

# =============================================================================
# HELPER
# =============================================================================

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"SAE: {SAE_RELEASE} ({SAE_WIDTH})")
    print(f"Layers: {TARGET_LAYERS}")
    print()
    print("Sample Sizes:")
    print(f"  Discovery: {N_SAMPLES_DISCOVERY}")
    print(f"  Train: {N_SAMPLES_DISCOVERY_TRAIN}")
    print(f"  Test: {N_SAMPLES_DISCOVERY_TEST}")
    print(f"  Eval: {N_SAMPLES_EVAL}")
    print()
    print(f"Languages: {list(LANGUAGES.keys())}")
    print(f"Target: {TARGET_LANGUAGE}")
    print()
    print("Thresholds:")
    print(f"  Monolinguality: M > {MONOLINGUALITY_THRESHOLD}")
    print(f"  Script detection: {SCRIPT_DETECTION_THRESHOLD}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
# Minimum data requirements for publication-grade runs
MIN_LANG_SAMPLES = int(os.environ.get("MIN_LANG_SAMPLES", "200"))
MIN_STEERING_PROMPTS = int(os.environ.get("MIN_STEERING_PROMPTS", "200"))
MIN_JUDGE_CALIB_PER_CLASS = int(os.environ.get("MIN_JUDGE_CALIB_PER_CLASS", "50"))
