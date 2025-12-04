"""Configuration for SAE multilingual steering experiments.

CORRECTED from methodology document:
- H1 falsification said "monolinguality >0.7" but Section 4.1 defines M>3.0 as strongly language-specific
- Using M>3.0 as the correct threshold per arXiv:2505.05111

UPDATES:
- Added LLM-as-judge configuration for semantic vs syntax evaluation
- Added Flash Attention 3 environment variables (for Hopper/H100)
- Updated FLORES dataset reference to openlanguagedata/flores_plus
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Model: Using Gemma 2 2B (fits comfortably on 40GB with SAEs)
MODEL_ID = "google/gemma-2-2b"
N_LAYERS = 26
HIDDEN_DIM = 2304

# Attention implementation: "sdpa" (default), "flash_attention_2", or "eager"
# For Flash Attention 3 on H100/Hopper, use environment variables before import:
#   export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
#   export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
#   ... (see flash_attn3_env_vars below)
ATTN_IMPLEMENTATION = "sdpa"  # Works on A100; use "flash_attention_2" if installed

# Flash Attention 3 environment variables for H100 (from Twitter compile guide)
# Set these BEFORE importing torch/transformers to compile FA3 in ~1-2 mins
FLASH_ATTN3_ENV_VARS = {
    "FLASH_ATTENTION_DISABLE_BACKWARD": "TRUE",
    "FLASH_ATTENTION_DISABLE_SPLIT": "TRUE",
    "FLASH_ATTENTION_DISABLE_LOCAL": "TRUE",
    "FLASH_ATTENTION_DISABLE_PAGEDKV": "TRUE",
    "FLASH_ATTENTION_DISABLE_FP16": "TRUE",
    "FLASH_ATTENTION_DISABLE_FP8": "TRUE",
    "FLASH_ATTENTION_DISABLE_APPENDKV": "TRUE",
    "FLASH_ATTENTION_DISABLE_VARLEN": "TRUE",
    "FLASH_ATTENTION_DISABLE_CLUSTER": "FALSE",  # Keep enabled
    "FLASH_ATTENTION_DISABLE_PACKGQA": "TRUE",
    "FLASH_ATTENTION_DISABLE_SOFTCAP": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM64": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM96": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM128": "FALSE",  # Keep enabled (Gemma uses 128)
    "FLASH_ATTENTION_DISABLE_HDIM192": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM256": "TRUE",
}

# ============================================================================
# SAE CONFIGURATION
# ============================================================================
# SAE: Gemma Scope 16k width (memory-efficient, JumpReLU architecture)
# Release ID format for sae-lens: gemma-scope-2b-pt-res-canonical
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_WIDTH = "16k"

# Layers to analyze: 40-60% of 26 layers = layers 10-16, plus some early/late for H3
TARGET_LAYERS = [5, 8, 10, 13, 16, 20, 24]

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================
# Languages (FLORES+ codes - ISO 639-3 + script)
# Updated for openlanguagedata/flores_plus dataset
LANGUAGES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
# Monolinguality threshold: CORRECTED to 3.0 per methodology Section 4.1
# M_j(L) = P(feature activates | lang L) / max P(feature activates | other lang)
# M > 3.0 = strongly language-specific (feature 3x more likely for target lang)
MONOLINGUALITY_THRESHOLD = 3.0

# Steering
STEERING_STRENGTHS = [0.5, 1.0, 2.0, 4.0, 8.0]
NUM_FEATURES_OPTIONS = [10, 25, 50]

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Traditional thresholds (for baseline comparison)
LABSE_THRESHOLD = 0.7
LID_CONFIDENCE_THRESHOLD = 0.5

# LLM-as-Judge configuration (for semantic vs syntax tie-breaking)
# Reference: MM-Eval (arXiv:2410.17578) for multilingual judge evaluation
# Using Gemini 2.5 Flash (FREE API!) - set GOOGLE_API_KEY environment variable
LLM_JUDGE_ENABLED = True
LLM_JUDGE_PROVIDER = "gemini"  # Options: "gemini", "claude", "openai"
LLM_JUDGE_MODEL = "gemini-2.5-flash"  # Free! Use "gemini-2.5-pro" for higher quality
LLM_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a language model's output is in the target language (Hindi).

Input prompt: {prompt}
Model output: {output}

Evaluate on two dimensions:
1. SCRIPT: Is the output written in Devanagari script? (Yes/No/Partial)
2. SEMANTIC: Is the content semantically appropriate Hindi? (Yes/No/Partial)
   - Check for grammatical Hindi structure
   - Check if it's meaningful Hindi vs transliteration/gibberish

Respond ONLY with a JSON object, no other text:
{{"script": "Yes/No/Partial", "semantic": "Yes/No/Partial", "explanation": "brief reason"}}
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
N_SAMPLES_DISCOVERY = 500  # sentences per language for feature discovery
N_SAMPLES_EVAL = 200  # sentences for steering evaluation


def setup_flash_attn3_env():
    """Set environment variables for Flash Attention 3 compilation on H100.
    
    Call this BEFORE importing torch/transformers.
    Only needed if using Flash Attention 3 (Hopper architecture).
    """
    for key, value in FLASH_ATTN3_ENV_VARS.items():
        os.environ[key] = value
    print("Flash Attention 3 environment variables set for H100 compilation")
