# Skeptical Deep Dive & Final Methodology Review

**Date:** 2025-12-09
**Auditor:** Skeptical Senior ML Researcher (Persona)

## 1. Executive Summary: "Mostly Solid, One Specific Leakage Risk"

I have performed a targeted, "adversarial" review of the `sae_multilingual` codebase against your specific list of doubts.

**The Verdict:** The project is scientifically stronger than 90% of SAE papers I review. The definitions of "detector" vs "generator" are operationalized well. The script/semantic split in Exp6/13 is a clever, defensible heuristic.

**The One Red Flag:** **Experiment 4 (Spillover) currently reuses the exact same FLORES English sentences for both steering vector construction and evaluation prompts.**
*   **Why strict reviewers will hate this:** You are steering the model on sentences $X$ and testing on $X$. The model might just be overfitting to the specific steering vector directions dominant in $X$.
*   **Fix:** I strongly recommend updating `exp4_spillover.py` to use `data.py::load_research_data()` (like Exp9 does), ensuring the steering vector is built from `train` and evaluation runs on `test` or `steering_prompts`.

---

## 2. Addressing Your Specific Doubts

### A. SAE vs Neuron-Level Interpretations (Detectors Peak Late)
*   **Doubt:** Are we over-interpreting SAE features?
*   **Analysis:** Your results (Exp1: detectors peak late, Exp2/9: generators peak mid-late) contradict some neuron-level literature (which finds language neurons in mid-layers).
*   **Defense:** This is a *feature* of your paper, not a bug. SAEs disentangle superposition. It is highly plausible that while *neurons* appear polysemantic or mixed in mid-layers, the *SAE features* resolving them show a different distribution.
*   **Reviewer-Proofing:** Frame this explicitly: *"Unlike neuron-level probing which finds mid-layer peaks, our SAE decomposition reveals that highly monosemantic language detectors concentrate in late layers, suggesting that language identity remains superposed or distributed in the 'messy middle' and is only resolved into sparse features prior to decoding."*

### B. Correlation vs Causation (Exp7/10/13)
*   **Doubt:** Is it still just correlation?
*   **Analysis:**
    *   **Exp7 (Causal Probing):** You effectively perform "do-calculus" on individual features ($do(f=f+\alpha)$). This *is* causal evidence.
    *   **Exp13 (Group Ablation):** You ablate entire sets defined by your heuristics. If the "script" set ablation kills script ratios but preserves semantics, that is strong causal evidence.
*   **Verdict:** You have done enough. You are not just looking at $M$ scores; you are intervening.

### C. Script vs Semantics Split (Heuristic validity)
*   **Doubt:** Is the definition (active on HI+UR+EN = semantic) too loose?
*   **Analysis:** It's a heuristic, but `Exp6` (transliteration control) is the killer validation. If features active on Devanagari *also* fire on Latin-transliterated Hindi, they *cannot* be purely visual/orthographic.
*   **Verdict:** Defensible. Keep the term "Operational Definition" prominent.

### D. FLORES EN Reuse (The Red Flag)
*   **Analysis:** In `exp4_spillover.py`, lines 330-345 load FLORES (first 200 rows) and use them for *both* `compute_steering_vector` and `prompts`.
*   **Severity:** **High.** Ideally, vectors should come from `train` split, prompts from `devtest`.
*   **Mitigation:** Update Exp4 to use the `DataSplit` class from `data.py`.

### E. Semantic Metrics (LaBSE)
*   **Doubt:** Is LaBSE good enough?
*   **Analysis:** LaBSE is the industry standard for bitext mining in Indic languages. It is robust for HI/BN/TA/TE. It might be weaker for low-resource (AS/OR), but your core claims rest on HI/UR.
*   **Verdict:** Acceptable. Add a caveat: *"LaBSE similarity is used as a proxy for semantic preservation; we acknowledge that for low-resource languages, embedding alignment may be imperfect."*

### F. Judge Calibration (Surrogate Ground Truth)
*   **Analysis:** You calibrate the Gemini judge against your `is_target_script` + `semantic_similarity` metric.
*   **Critique:** You are essentially training the judge to agree with your regex/LaBSE pipeline.
*   **Defense:** This is still valuable. It ensures the judge isn't hallucinating "High Quality Hindi" on garbage text that LaBSE hates. It aligns the "soft" judge with "hard" metrics.
*   **Verdict:** Valid, but rename "accuracy" to "Alignment with Hard Metrics" in your head (and maybe the text).

---

## 3. Recommended Actions

1.  **Refactor Exp4:** Switch `exp4_spillover.py` to use `load_research_data`.
2.  **Run Full Suite:** Execute the run script.
3.  **Final Polish:** Ensure `paper.tex` explicitly contrasts your SAE results with prior neuron-level work (Point A).

This project is in excellent shape. The skepticism is healthy, but don't let it paralyze you—you have the experiments to back up your claims.
