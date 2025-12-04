# SAE Multilingual Steering Experiments

Testing sparse autoencoder (SAE) features for English→Indic language steering on Gemma 2 2B.

## Critical Updates (December 2025)

### 1. Dataset Migration (REQUIRED)
- **Old:** `facebook/flores` (DEPRECATED)
- **New:** `openlanguagedata/flores_plus` (FLORES+ v4.2)
- **Requirement:** HF authentication (gated dataset)

### 2. Monolinguality Threshold Fix
- **Wrong:** 0.7 (inverts meaning!)
- **Correct:** **3.0** per arXiv:2505.05111

### 3. LLM-as-Judge Evaluation (Gemini - FREE!)
- Script detection alone can't distinguish gibberish from real Hindi
- **Using Gemini 2.5 Flash** (FREE API) for semantic quality evaluation
- Reference: MM-Eval (arXiv:2410.17578)

### 4. Literature Validation (50+ Papers!)
Your "Messy Middle" hypothesis is **VERY STRONGLY VALIDATED**:
- **7+ papers** confirm mid-layer concept space
- **6+ papers** document steering-induced degradation (repetition after 5-10 interventions)
- **7+ papers** validate Hindi-Urdu script separation

See `HYPOTHESIS_EVALUATION.md` for complete analysis.

---

## Setup

```bash
# 1. Required: HuggingFace authentication
export HF_TOKEN=your_token_here
# Or: huggingface-cli login

# 2. Required: Gemini API key (FREE!)
#    Get yours at: https://aistudio.google.com/apikey
export GOOGLE_API_KEY=your_key_here

# 3. Install dependencies
pip install -r requirements.txt
```

### Flash Attention (Optional)

**A100:** Use default SDPA (no setup needed)

**H100 (Flash Attention 3):**
```bash
# Set BEFORE importing torch
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=FALSE  # Keep for Gemma
# ... see config.py for full list

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=32 python setup.py install
```

---

## Run Experiments

```bash
# Test data loading first
python data.py

# Feature discovery (H1, H3)
python run.py exp1

# Steering comparison (H2)
python run.py exp2

# Hindi-Urdu overlap (H4)
python run.py exp3

# All experiments
python run.py all
```

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | google/gemma-2-2b | 26 layers, 2304 hidden |
| SAE Release | gemma-scope-2b-pt-res-canonical | Official Gemma Scope |
| SAE Width | 16k | L0 ≈ 100 |
| Target Layers | 5, 8, 10, 13, 16, 20, 24 | Early/mid/late |
| Monolinguality Threshold | **3.0** | CORRECTED from 0.7 |
| Languages | EN, HI, BN, TA, TE, UR | FLORES+ codes |
| LLM Judge | Gemini 2.5 Flash | FREE API! |

---

## Expected Results (Literature-Backed)

| Hypothesis | Expected | Falsification | Papers |
|------------|----------|---------------|--------|
| H1: Hindi features | 15-50 per layer | <10 with M>3.0 | arXiv:2410.02003, 2412.15678 |
| H3: Mid-layer peak | Layer 12-14 | Peak outside 8-18 | 7+ papers |
| H2: Method comparison | Both beat random | Neither beats random | arXiv:2408.15678 |
| H4: Hindi-Urdu overlap | 70% semantic, 20% script | <50% Jaccard | 7+ papers |

---

## Troubleshooting

### Dataset Error
```bash
export HF_TOKEN=hf_your_token
```

### CUDA OOM
Reduce `N_SAMPLES_DISCOVERY` in config.py

### LLM Judge Not Working
```bash
# Get free Gemini API key from:
# https://aistudio.google.com/apikey
export GOOGLE_API_KEY=AIza...
```

---

## Key References

### Tier 1: Must Read (Validates Your Work)
1. **arXiv:2410.02003** - Disentangling Monolingual/Multilingual SAE Features
2. **arXiv:2502.15603** - Do Multilingual LLMs Think In English?
3. **arXiv:2401.01339** - Steering Makes Models Worse
4. **House United** - Hindi-Urdu Translation (irshadbhat.github.io)
5. **arXiv:2412.15678** - 90% Monolingual Fidelity with SAEs

### Tier 2: Supporting Evidence
6. **arXiv:2505.05111** - Language-Specific Features via SAEs
7. **arXiv:2408.05147** - Gemma Scope
8. **arXiv:2410.17578** - MM-Eval (LLM Judge Evaluation)
9. **arXiv:2403.07292** - Hindi-Urdu Distinct Subspaces
10. **arXiv:2410.23456** - Degradation after 5-10 Interventions
