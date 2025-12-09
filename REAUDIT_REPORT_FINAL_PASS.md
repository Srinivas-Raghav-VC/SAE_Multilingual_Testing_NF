# Final Audit Report

**Date:** 2025-12-09
**Auditor:** Gemini (Skeptical Reviewer Mode)
**Verdict:** **PASS (Methodologically Sound)**

I have completed the final re-audit as requested in `GEMINI_3PRO_FINAL_CHECK.md`. The critical data leakage issues identified in the previous "Conditional Pass" have been successfully resolved.

---

## 1. Verification of Fixes

### 1.1 Data Leakage & Train/Test Separation
**Status: FIXED**

*   **`experiments/exp2_steering.py`**:
    *   **Vector Construction**: Now uses `data_split.train['en']` and `data_split.train['hi']` (derived from Samanantar/FLORES-train).
    *   **Evaluation**: Now uses `data_split.steering_prompts` (derived from EVAL_PROMPTS + FLORES-devtest).
    *   **Separation**: `load_research_data()` explicitly deduplicates train vs test/prompts. This eliminates the "training on test" bug.

*   **`experiments/exp7_causal_feature_probing.py`**:
    *   **Candidate Selection**: Now uses `data_split.train['en']` / `train['hi']`.
    *   **Probing Prompts**: Now uses `data_split.steering_prompts`.
    *   **Separation**: Verified. The candidate features are selected from one distribution, and their causal effects are measured on a held-out distribution.

*   **`data.py` Integrity**:
    *   The `load_research_data` function includes explicit deduplication logic: `train_data[lang] = [s for s in train_data[lang] if s not in test_set]`. This ensures the integrity of the split even when using FLORES for both training (fallback) and testing.

### 1.2 Core Infrastructure
**Status: SOLID**

*   **`model.py`**: Steering hooks are robust (checking bounds, device, dtype, and zero-norm).
*   **`evaluation_comprehensive.py`**:
    *   Judge calibration correctly implements the Lee et al. bias correction (`(p+q0-1)/(q0+q1-1)`).
    *   LaBSE usage is safe (truncation + warning).
    *   Metrics (Jaccard, Script Ratio) are mathematically correct.
*   **Main Experiments (Exp4, Exp9, Exp11, Exp12)**:
    *   These scripts consistently use `load_research_data()`, ensuring a unified and rigorous train/test methodology across the entire project.

---

## 2. Final Verdict

**Verdict: PASS**

The codebase now adheres to high standards of statistical hygiene. The previous critical flaws (data leakage in sanity checks) are fixed. The project is ready for a strong submission.

*   **Methodological Soundness**: **High**. The use of a "calibrated judge" and rigorous data splitting sets this apart from typical amateur interpretability work.
*   **Statistical Adequacy**: **Good**. Sample sizes (N≈100-200 for eval, N≈5000 for vectors) are sufficient for the claims made.
*   **Leakage**: **None Detected**.

### Remaining Minor Concerns (For Paper Caveats)

While the code is fixed, transparently addressing these in the paper will "reviewer-proof" it:

1.  **Surrogate Ground Truth**: You calibrate the LLM judge against a "structural" metric (Script + LaBSE). Acknowledging this as a "proxy" rather than "human gold standard" is honest and safe.
2.  **Causal Claims (Exp7)**: Even with clean data, Exp7 is "causal" in the *model's* sense (intervention) but relies on *observational* feature selection. Frame it as "Interventional verification of candidate features" rather than "Proof of mechanism."
3.  **9B Scaling**: Keep this as a "preliminary scaling check" rather than a central contribution, given the lighter analysis compared to 2B.

---

## 3. Pre-Submission Checklist

### 3.1 Recommended Execution Order

Since the code has changed, re-run in this order to ensure all results are fresh and consistent:

1.  **`python experiments/exp2_steering.py`** (Sanity check: ensures steering still works without leakage)
2.  **`python experiments/exp9_layer_sweep_steering.py`** (**CORE**: The main results table)
3.  **`python experiments/exp11_judge_calibration.py`** (Calibrate the judge for Exp9/12)
4.  **`python experiments/exp12_qa_degradation.py`** (QA performance)
5.  **`python experiments/exp7_causal_feature_probing.py`** (Mechanism check)
6.  **`python experiments/exp4_spillover.py`** (Clustering analysis)
7.  **`python summarize_results.py`** & **`python plots.py`**

### 3.2 Core vs. Appendix Claims

*   **Main Text**:
    *   **Exp9**: Layer-wise steering effectiveness (Heatmaps).
    *   **Exp1/3**: Feature geometry (Detector vs Generator counts).
    *   **Exp12/11**: QA degradation & Calibrated Judge results.
    *   **Exp4**: Language clustering/spillover (Strong evidence of structure).

*   **Appendix**:
    *   **Exp2**: Method comparison (mostly subsumed by Exp9, good for "we checked baselines").
    *   **Exp7**: Causal probing (supporting evidence for mechanism).
    *   **Exp15/16**: Symmetry & Code-mixing (Exploratory).
    *   **Exp8**: 9B Scaling (Preliminary).

### 3.3 Top Residual Risks (Pre-empt these!)

1.  **"Did you just memorize FLORES?"** -> **Rebuttal**: "No, we strictly deduplicated training vectors from evaluation prompts (Section X.X)."
2.  **"Is the Judge reliable for Urdu?"** -> **Rebuttal**: "We applied Lee et al.'s bias correction, estimating specificity/sensitivity on a held-out calibration set (Figure Y)."
3.  **"Why LaBSE?"** -> **Rebuttal**: "LaBSE is the standard open embedding model for Indic languages; we truncate to avoid artifacts."

**Congratulations. This is now a rigorous piece of research.**
