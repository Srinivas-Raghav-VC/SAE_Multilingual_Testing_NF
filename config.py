"""Unified Configuration for SAE Multilingual Steering Experiments.

This config provides backward compatibility with existing experiment files
while using research-grade parameters.

Variables are named for compatibility with:
- exp1_feature_discovery.py
- exp2_steering.py  
- exp3_hindi_urdu.py
"""

import os
from typing import List, Dict

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME = "google/gemma-2-2b"
MODEL_ID = MODEL_NAME  # Alias for backward compatibility
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
N_LAYERS = 26
HIDDEN_DIM = 2304
SAE_WIDTH = 16384
ATTN_IMPLEMENTATION = "flash_attention_2"  # or "eager" if no flash attention

# For 9B model (set USE_9B_MODEL = True to switch)
USE_9B_MODEL = False
if USE_9B_MODEL:
    MODEL_NAME = "google/gemma-2-9b"
    MODEL_ID = MODEL_NAME
    SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
    N_LAYERS = 42
    HIDDEN_DIM = 3584
    SAE_WIDTH = 131072

# ============================================================================
# SAMPLE SIZES (Research-Grade)
# ============================================================================
# For backward compatibility - maps to new names
N_SAMPLES_DISCOVERY = 5000          # Total for discovery
N_SAMPLES_DISCOVERY_TRAIN = 4000    # 80% for training
N_SAMPLES_DISCOVERY_TEST = 1000     # 20% for testing
N_SAMPLES_EVAL = 200                # For steering evaluation
N_STEERING_EVAL = 200               # Alias

# Train/test split
TRAIN_TEST_SPLIT = 0.8

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================
LANGUAGES = ["en", "hi", "ur", "bn", "ta", "te"]
TARGET_LANGUAGE = "hi"

# Extended languages for generalization testing
INDIC_LANGUAGES = ["hi", "bn", "ta", "te", "mr", "gu", "pa", "ml"]
NON_INDIC_LANGUAGES = ["de", "fr", "zh", "ja", "ar", "ru"]

# FLORES+ language code mapping
FLORES_LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ml": "mal_Mlym",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
}

# ============================================================================
# LAYER CONFIGURATION
# ============================================================================
# Target layers for analysis (covers early, mid, late)
TARGET_LAYERS = [5, 8, 10, 13, 16, 20, 24]

# Layer ranges for categorization
LAYER_RANGES = {
    "early": [3, 5, 7],
    "middle": [10, 13, 15],
    "late": [18, 21, 24],
}

# Mid-range for hypothesis testing (40-60% of depth)
MID_RANGE = (10, 16)

# ============================================================================
# FEATURE SELECTION THRESHOLDS
# ============================================================================
# Monolinguality threshold (M > threshold means language-specific)
MONOLINGUALITY_THRESHOLD = 3.0

# Activation threshold for "active" features
ACTIVATION_THRESHOLD = 0.1

# Number of features to select for steering
N_STEERING_FEATURES = 25
NUM_FEATURES = 25  # Alias for backward compatibility
NUM_FEATURES_OPTIONS = [10, 25, 50, 100]  # Options to test

# ============================================================================
# STEERING CONFIGURATION
# ============================================================================
# Steering strengths to evaluate
STEERING_STRENGTHS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]

# Methods to compare
STEERING_METHODS = ["activation_diff", "monolinguality", "random", "dense"]

# ============================================================================
# EVALUATION THRESHOLDS
# ============================================================================
# Script detection (success if >= threshold Devanagari)
SCRIPT_DETECTION_THRESHOLD = 0.3
DEVANAGARI_THRESHOLD = 0.3  # Alias

# Repetition thresholds (degradation indicators)
REPETITION_3GRAM_THRESHOLD = 0.3
REPETITION_5GRAM_THRESHOLD = 0.2

# Perplexity increase threshold
PERPLEXITY_INCREASE_THRESHOLD = 2.0

# ============================================================================
# LLM-AS-JUDGE CONFIGURATION
# ============================================================================
LLM_JUDGE_PROVIDER = "gemini"  # Options: "gemini", "claude", "openai"
LLM_JUDGE_MODEL = "gemini-2.0-flash"  # FREE!

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
# EVALUATION PROMPTS
# ============================================================================
# Prompts for steering evaluation (kept simple for backward compatibility)
EVAL_PROMPTS = [
    "Hello, how are you today?",
    "What is your name?",
    "Tell me about yourself.",
    "What is the weather like?",
    "How can I help you?",
    "What time is it?",
    "Where do you live?",
    "What is your favorite food?",
    "Tell me a story.",
    "What do you think about life?",
    "Once upon a time, there was a",
    "The story begins with",
    "In a small village,",
    "Long ago, in a distant land,",
    "The sun rose over the",
    "Write a poem about nature.",
    "Describe a beautiful sunset.",
    "Tell me about Indian culture.",
    "What is the meaning of happiness?",
    "How do you learn new things?",
]

# Extended prompts for research-grade evaluation
EXTENDED_EVAL_PROMPTS = EVAL_PROMPTS + [
    "What is your hobby?",
    "Do you have any siblings?",
    "What is your favorite color?",
    "Where are you from?",
    "What did you eat today?",
    "What are you thinking about?",
    "Can you help me?",
    "What is the date today?",
    "How do you feel?",
    "What makes you happy?",
    "What is your dream?",
    "Tell me a joke.",
    "What is love?",
    "Why is the sky blue?",
    "What is the meaning of life?",
    "How does a computer work?",
    "What is artificial intelligence?",
    "Who is your hero?",
    "What is your goal?",
    "What do you want to learn?",
    "How can I improve myself?",
    "What is success?",
    "What is failure?",
    "How do you learn new things?",
    "What is creativity?",
    "What is wisdom?",
    "How do you solve problems?",
    "What is friendship?",
    "What is family?",
    "What is culture?",
    "What is tradition?",
    "Explain how to make tea.",
    "Describe the steps to plant a tree.",
    "Tell me how to cook rice.",
    "Explain how to read a book.",
    "Describe how to write a letter.",
    "Tell me how to exercise.",
    "Explain how to save money.",
    "Describe how to stay healthy.",
    "Tell me how to learn a language.",
    "Explain how to be a good friend.",
]

# Capability benchmark prompts
CAPABILITY_BENCHMARK_PROMPTS = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
    "How many continents are there?",
    "What planet is known as the Red Planet?",
    "If I have 3 apples and give away 1, how many do I have?",
    "What comes next in the sequence: 2, 4, 6, 8, ?",
    "If all dogs are animals, and Buddy is a dog, is Buddy an animal?",
    "What is 15 + 27?",
    "If today is Monday, what day is tomorrow?",
    "List three colors.",
    "Write a word that rhymes with 'cat'.",
    "Name one country in Europe.",
    "What is the opposite of 'hot'?",
    "Count from 1 to 5.",
]

# ============================================================================
# SCRIPT UNICODE RANGES
# ============================================================================
SCRIPT_UNICODE_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "arabic": (0x0600, 0x06FF),
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "latin": (0x0041, 0x007A),
    "cyrillic": (0x0400, 0x04FF),
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
CHECKPOINTS_DIR = "checkpoints"

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ============================================================================
# COMPUTATIONAL SETTINGS
# ============================================================================
BATCH_SIZE = 8
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
USE_FLASH_ATTENTION = True

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEEDS = [42, 123, 456]
DEFAULT_SEED = 42
SEED = 42  # Alias

# ============================================================================
# PRINT CONFIGURATION ON IMPORT (optional)
# ============================================================================
def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"SAE Release: {SAE_RELEASE}")
    print(f"Layers: {N_LAYERS}")
    print(f"Target Layers: {TARGET_LAYERS}")
    print(f"")
    print(f"Sample Sizes:")
    print(f"  Discovery: {N_SAMPLES_DISCOVERY}")
    print(f"  Train: {N_SAMPLES_DISCOVERY_TRAIN}")
    print(f"  Test: {N_SAMPLES_DISCOVERY_TEST}")
    print(f"  Eval: {N_SAMPLES_EVAL}")
    print(f"")
    print(f"Languages: {LANGUAGES}")
    print(f"Target: {TARGET_LANGUAGE}")
    print(f"")
    print(f"Thresholds:")
    print(f"  Monolinguality: M > {MONOLINGUALITY_THRESHOLD}")
    print(f"  Script detection: {SCRIPT_DETECTION_THRESHOLD}")
    print(f"")
    print(f"LLM Judge: {LLM_JUDGE_PROVIDER} ({LLM_JUDGE_MODEL})")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
