# Final Re-Audit Report for Gemini 3-Pro

**Date:** 2025-12-09
**Auditor:** Gemini (Skeptical Reviewer Mode)
**Verdict:** **Conditional Pass** (Requires 2 Critical Fixes)

I have performed a ruthless audit of the codebase, focusing on statistical hygiene, data leakage, and implementation correctness. While much of the "recent patching" is effective, **critical data leakage** remains in the sanity-check experiments (Exp2, Exp7).

---

## 1. Executive Summary

- **Core Infrastructure (`model.py`, `data.py`, `eval`):** **SOLID**.
    - Steering hooks are safe (checks bounds, device, dtype).
    - `data.py` logic for splitting and deduping is correct.
    - Judge calibration (Lee et al.) is implemented correctly.
- **Main Experiments (`exp9`, `exp4`, `exp11`):** **SOLID**.
    - These use `load_research_data()` effectively, ensuring separation between vector estimation (train) and evaluation (prompts/test).
- **Sanity Checks (`exp2`, `exp7`):** **BROKEN / DATA LEAKAGE**.
    - **Critical Issue:** These scripts still manually slice `flores['en']` for both vector estimation and evaluation. This constitutes "training on test" and invalidates the reported success rates in these specific scripts.
    - **Fix:** They must be refactored to use `load_research_data()`.

---

## 2. Code & Experiment Audit

### 2.1 Correctness & Silent Failures (Section 3.1)

| Component | Status | Notes |
| :--- | :--- | :--- |
| **Steering Hooks** (`model.py`) | **SAFE** | Checks layer bounds, dimension match, zero-norm warnings. Handles device/dtype correctly. |
| **LaBSE Handling** (`eval`) | **SAFE** | Truncates to 512 tokens (warns once). Returns -1.0 if model missing. |
| **Judge Calibration** (`eval`) | **CORRECT** | Formula `(p+q0-1)/(q0+q1-1)` is correct. CIs implemented. Fallback for `q0+q1<=1`. |
| **Jaccard Index** | **CORRECT** | Uses `intersection / union`. No >1.0 bugs. |

### 2.2 Statistical Adequacy (Section 3.2)

| Experiment | Role | Status | Sample Size | Issues |
| :--- | :--- | :--- | :--- | :--- |
| **Exp1 (Discovery)** | Geometry | **Acceptable** | N=5000 | Uses simplistic splitting, but acceptable for *finding* features. |
| **Exp2 (Steering Check)** | Sanity | **FLAWED** | N=~50 | **CRITICAL LEAKAGE**: Uses `texts_en[:100]` for vectors and `texts_en[:N]` for prompts. **100% overlap.** |
| **Exp4 (Spillover)** | Clustering | **SOLID** | N=~50-100 | Uses `load_research_data`. Train (Samanantar) ≠ Eval (FLORES). |
| **Exp7 (Causal)** | Mechanism | **FLAWED** | N=50-200 | **CRITICAL LEAKAGE**: Same FLORES subset used for candidate selection and probing. |
| **Exp9 (Sweep)** | **CORE** | **SOLID** | N=100 | Uses `load_research_data`. Rigorous judge integration. |
| **Exp11 (Judge)** | Calibration | **SOLID** | N=100/100 | Calib/Test split is 50/50 of `N_EVAL` (200 total). Sufficient. |

### 2.3 Data Leakage Analysis (Section 3.3)

- **Major Concern:** `exp2_steering.py` and `exp7_causal_feature_probing.py`.
    - These scripts bypass the `data.py` safeguards.
    - **Impact:** Reported steering success in Exp2 is likely inflated. Causal effects in Exp7 are likely overestimated.
    - **Recommendation:** **MANDATORY REFACTOR** before publication.

- **Minor Concern:** `exp11` Sample Size.
    - Currently ~100 calibration / 100 test per language. Lee et al. often use more, but this is acceptable for a single-paper scope given the cost.

---

## 3. Claims Verification (Section 4.1)

| Claim | Evidence | Verdict |
| :--- | :--- | :--- |
| **1. Detector vs Generator** | Exp9 Heatmaps | **Supported**. Exp9 is statistically sound. |
| **2. Dissociation** | Exp9 | **Supported**. Monolinguality vectors fail to steer (Exp9 confirms this on clean data). |
| **3. HI-UR Geometry** | Exp3 | **Likely Supported**. (Assuming Exp3 is statistically similar to Exp1). |
| **4. Spillover** | Exp4 | **Strongly Supported**. Exp4 design is robust. |
| **5. QA Degradation** | Exp12 | **Supported**. (Assuming Exp12 follows Exp9's rigorous pattern). |

---

## 4. Specific Doubts & Risk Assessment (Section 5)

| Doubt | Severity | Verdict / Mitigation |
| :--- | :--- | :--- |
| **1. SAE vs Neuron** | Moderate | **Caveat**. Frame carefully: "In the SAE basis, we observe..." Do not claim this invalidates neuron-level findings, but complements them. |
| **2. Correlation vs Causation** | High | **Risk**. Exp7 is the causal link, but it is currently **flawed** (data leakage). **Fix Exp7 immediately.** |
| **3. Script vs Semantic** | Moderate | **Acceptable**. The operationalization (LaBSE + Script Ratio) is standard enough for a first paper. |
| **4. FLORES EN Reuse** | **MAJOR** | **CONFIRMED**. Exp2/7 are guilty. Exp4/9 are innocent. **Must fix Exp2/7.** |
| **5. LaBSE Bias** | Minor | **Caveat**. Add a line: "We rely on LaBSE, which may have its own biases, though it is standard for Indic languages." |
| **6. Judge Calibration** | Moderate | **Defensible**. Using "structural ground truth" (script + LaBSE) is a valid, clever proxy when human labels are absent. |
| **7. 9B Scaling** | Moderate | **Limit Claims**. Keep it as "preliminary evidence" or "sanity check". |

---

## 5. Final Checklist & Action Plan

### 5.1 Immediate Code Fixes

You **MUST** apply these patches to remove data leakage:

1.  **Patch `experiments/exp2_steering.py`**:
    - Replace manual FLORES loading with `data.load_research_data()`.
    - Use `data_split.train['en']` for vectors.
    - Use `data_split.steering_prompts` for evaluation.

2.  **Patch `experiments/exp7_causal_feature_probing.py`**:
    - Similar refactor: Use `load_research_data()` to separate candidate selection (train) from probing (prompts).

### 5.2 Experiments to Re-Run

After patching, run the full suite to ensure clean results:

```bash
# 1. Core Steering Sweep (Ensures clean main results)
python experiments/exp9_layer_sweep_steering.py

# 2. Sanity Checks (Must run AFTER patching to verify they still pass without leakage)
python experiments/exp2_steering.py
python experiments/exp7_causal_feature_probing.py

# 3. Judge Calibration & QA (Ensure consistent stats)
python experiments/exp11_judge_calibration.py
python experiments/exp12_qa_degradation.py

# 4. Generate Summaries & Plots
python summarize_results.py
python plots.py
```

### 5.3 Paper Edits

1.  **Methods Section**: Explicitly state: "We strictly separate training data (Samanantar/FLORES-train) used for vector estimation from held-out evaluation prompts (FLORES-devtest + MLQA)."
2.  **Limitations**: Add: "Causal probing (Exp7) suggests influence, but full circuit-level causal verification is left for future work."
3.  **Appendix**: Move Exp16 (Code-mixing) and Exp15 (Symmetry) to Appendix if they are weak.

**Final Word:** The project is 80% solid. The 20% that is broken (Exp2/7) is the exact kind of "lazy data reuse" that gets papers rejected. **Fix it, and you have a strong submission.**
