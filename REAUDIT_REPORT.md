# Ruthless Re-Audit Report: SAE Multilingual Steering Project

**Date:** 2025-12-09
**Auditor:** Skeptical Senior ML Researcher

## Executive Summary

I have performed a full code and experiment audit of the `sae_multilingual` repository. The codebase is generally high-quality and scientifically rigorous, with careful handling of activation rates, monolinguality scores, and cross-lingual overlaps.

**CRITICAL FINDING:** The most significant issue was a severe sample-size deficit in the steering evaluation and judge calibration experiments (`Exp9`, `Exp11`). The code relied on a small handwritten list of ~25 prompts (`EVAL_PROMPTS`), rendering statistical calibration of the LLM judge impossible (requires n ~ 100).

**STATUS:** I have **FIXED** this by modifying `data.py` to automatically augment the steering prompts with held-out FLORES English sentences up to `N_STEERING_EVAL` (200).

---

## Detailed Audit Findings

### 1. Statistical Adequacy & Sample Sizes

*   **Issue:** `Exp9` (Sweep) and `Exp11` (Calibration) used `data_split.steering_prompts`, which contained only 25 items. `Exp11` split this into ~12 calibration / ~13 test samples. This is statistically meaningless for estimating sensitivity/specificity.
*   **Fix:** Modified `load_research_data` in `data.py`. It now detects if `steering_prompts` is small and appends sentences from FLORES `devtest` (English) to reach 200 samples.
*   **Result:** `Exp11` will now have ~100 calibration and ~100 test samples, satisfying the requirements for the Lee et al. (2025) bias correction method. `Exp2` and `Exp7` already had local logic to use FLORES; `Exp4` also did. Now all experiments share a robust prompt set.

### 2. Silent Failures & Safety

*   **Issue:** `model.py` had no check for zero-norm steering vectors. If feature selection failed (e.g. no features met the criteria), the code might produce a zero vector, leading to "steered" generation that is identical to baseline, potentially confusing the analysis.
*   **Fix:** Added a check in `generate_with_steering` in `model.py`. It now issues a `UserWarning` if the steering vector norm is `< 1e-6`.
*   **Verified:** `evaluation_comprehensive.py` handles missing Gemini keys gracefully (`is_gemini_available` check). `LaBSE` truncation is handled (though crude split-by-whitespace is used, it prevents crashes).

### 3. Data Leakage

*   **Check:** `train` (Samanantar/FLORES train) vs `test` (FLORES devtest).
*   **Finding:** `data.py` explicitly removes any sentence found in `test` from `train`. This is rigorous.
*   **Reuse:** FLORES English sentences are used for both "activation difference" vector construction (train split) and "steering prompts" (test split). This is acknowledged in `paper.tex` and is standard practice (vector estimation vs generation are different modalities).

### 4. Judge Calibration Implementation

*   **Check:** `evaluation_comprehensive.py` implements Lee et al. (2025).
*   **Finding:** The equations for $\hat{\theta}$ and confidence intervals are correct. The logic handles cases where the judge is worse than random (though rare).
*   **Note:** The calibration relies on the "ground truth" being the *script+semantic* evaluator. This is a valid "surrogate ground truth" for measuring the judge's alignment with our automated metrics.

### 5. Paper & Plots

*   **`paper.tex`:** Accurately reflects the methodology. The "N/A" placeholders are appropriate. The distinction between "Detector" (monolinguality) and "Generator" (steering) features is well-articulated.
*   **`plots.py`:** Covers all key hypotheses (H1-H4, H8-H9). Logic for heatmaps and comparisons looks correct.

---

## Next Steps for the User

1.  **Run Experiments:** You can now safely run the full suite.
    ```bash
    python run.py --all
    ```
    (Or run individual experiments like `exp11_judge_calibration.py` first to ensure the judge is ready).

2.  **Generate Results:**
    ```bash
    python summarize_results.py
    python plots.py
    ```

3.  **Finalize Paper:**
    *   Replace "N/A" in `paper.tex` with values from `results/summary_report.txt`.
    *   Compile `paper.tex`.

The code is now **Submission-Ready** from a methodological and statistical standpoint.
