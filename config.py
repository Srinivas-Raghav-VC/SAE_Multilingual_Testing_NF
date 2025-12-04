"""Configuration for SAE multilingual steering experiments.

CORRECTED from methodology document:
- H1 falsification said "monolinguality >0.7" but Section 4.1 defines M>3.0 as strongly language-specific
- Using M>3.0 as the correct threshold per arXiv:2505.05111
"""

# Model: Using Gemma 2 2B (fits comfortably on 40GB with SAEs)
MODEL_ID = "google/gemma-2-2b"
N_LAYERS = 26
HIDDEN_DIM = 2304

# SAE: Gemma Scope 16k width (memory-efficient)
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_WIDTH = "16k"

# Layers to analyze: 40-60% of 26 layers = layers 10-16, plus some early/late for H3
TARGET_LAYERS = [5, 8, 10, 13, 16, 20, 24]

# Languages (FLORES-200 codes)
LANGUAGES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}

# Monolinguality threshold: CORRECTED to 3.0 per methodology Section 4.1
# M_j(L) = P(feature activates | lang L) / max P(feature activates | other lang)
# M > 3.0 = strongly language-specific (feature 3x more likely for target lang)
MONOLINGUALITY_THRESHOLD = 3.0

# Steering
STEERING_STRENGTHS = [0.5, 1.0, 2.0, 4.0, 8.0]
NUM_FEATURES_OPTIONS = [10, 25, 50]

# Evaluation thresholds
LABSE_THRESHOLD = 0.7
LID_CONFIDENCE_THRESHOLD = 0.5

# Data
N_SAMPLES_DISCOVERY = 500  # sentences per language for feature discovery
N_SAMPLES_EVAL = 200  # sentences for steering evaluation
