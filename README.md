# SAE Multilingual Steering Experiments

Testing sparse autoencoder (SAE) features for English→Indic language steering.

## Critical Hypothesis Evaluation

### H1: SAEs contain ≥10 robust Hindi-specific features per layer

**Status:** Testable

**Definition:** A feature is "Hindi-specific" if monolinguality M > 3.0, where:
```
M_j(Hindi) = P(feature_j activates | Hindi text) / max P(feature_j activates | other lang)
```

**NOTE:** The original methodology document incorrectly listed M > 0.7 as the threshold. This was corrected to M > 3.0 per arXiv:2505.05111 Section 4.1 definitions.

**Falsification:** <10 features with M > 3.0 in any target layer

---

### H2: Attribution-selected features outperform activation-selected by ≥5%

**Status:** Modified

**Critical Issue:** True "attribution" requires paired completions from the same prompt:
- Prompt: "The capital of France is"  
- Positive completion: "पेरिस" (Hindi)
- Negative completion: "Paris" (English)

This requires the model to naturally produce both languages from the same prompt, which is impractical.

**Modified Approach:** We compare:
1. **Activation-diff:** Features with largest activation difference between Hindi and English parallel texts
2. **Monolinguality:** Features with highest monolinguality score
3. **Random:** Random baseline
4. **Dense:** Mean difference in hidden space (no SAE decomposition)

**Metric:** Language shift rate (proportion of generations containing Devanagari script)

---

### H3: Mid-layers (40-60% depth) contain most language-specific features

**Status:** Testable

For Gemma 2 2B (26 layers): 40-60% = layers 10-16

**Falsification:** Peak feature density outside layers 10-16

---

### H4: Hindi-Urdu share >50% semantic features

**Status:** Testable

Hindi (Devanagari script) and Urdu (Arabic script) are the same spoken language. We test:
- **Script features:** Activate for one script but not the other
- **Semantic features:** Activate similarly for both (same content, different script)

**Data:** FLORES-200 parallel sentences in Hindi and Urdu

**Falsification:** <50% of active features are shared between Hindi and Urdu

---

## Setup

```bash
pip install -r requirements.txt
```

Requires A100 40GB GPU.

## Run Experiments

```bash
# Feature discovery (H1, H3)
python run.py exp1

# Steering comparison (H2)
python run.py exp2

# Hindi-Urdu overlap (H4)
python run.py exp3

# All experiments
python run.py all
```

## Project Structure

```
sae_multilingual/
├── config.py           # Configuration (corrected thresholds)
├── data.py             # FLORES-200 loading
├── model.py            # Gemma 2 2B + Gemma Scope SAE loading
├── run.py              # Main runner
├── requirements.txt
└── experiments/
    ├── exp1_feature_discovery.py  # H1, H3
    ├── exp2_steering.py           # H2
    └── exp3_hindi_urdu.py         # H4
```

## Model Configuration

- **Model:** Gemma 2 2B (google/gemma-2-2b)
- **SAEs:** Gemma Scope 16k width (gemma-scope-2b-pt-res-canonical)
- **Layers analyzed:** 5, 8, 10, 13, 16, 20, 24
- **Languages:** English, Hindi, Bengali, Tamil, Telugu, Urdu

Using Gemma 2 2B (not 9B) to fit comfortably on A100 40GB with SAE overhead.
