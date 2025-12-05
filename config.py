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
MODEL_ID = "google/gemma-2-2b"
MODEL_NAME = MODEL_ID  # Alias
N_LAYERS = 26
HIDDEN_DIM = 2304

# SAE Configuration
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_WIDTH = "16k"

# Attention implementation
ATTN_IMPLEMENTATION = "eager"  # or "eager"

# =============================================================================
# TARGET LAYERS
# =============================================================================
TARGET_LAYERS = [5, 8, 10, 13, 16, 20, 24]

LAYER_RANGES = {
    "early": [3, 5, 7],
    "middle": [10, 13, 15],
    "late": [18, 21, 24],
}

# =============================================================================
# LANGUAGES
# =============================================================================

# FLORES-200 codes
LANGUAGES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
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
}

TARGET_LANGUAGE = "hi"

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

# =============================================================================
# STEERING CONFIGURATION
# =============================================================================

STEERING_STRENGTHS = [0.5, 1.0, 2.0, 4.0, 8.0]
NUM_FEATURES = 25
NUM_FEATURES_OPTIONS = [10, 25, 50]

STEERING_METHODS = ["activation_diff", "monolinguality", "random", "dense"]

# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

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
# UNICODE RANGES FOR SCRIPT DETECTION
# =============================================================================

SCRIPT_RANGES = {
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
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

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
