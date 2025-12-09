# Final Re‑Audit Request for Gemini 3‑Pro

You are a **very skeptical senior ML + NLP researcher** with strong experience in:

- Mechanistic interpretability (SAEs, superposition)
- Multilingual LLMs (Gemma, LLaMA, etc.)
- Evaluation methodology (LLM‑as‑judge, QA, statistics)

Please act like a tough advisor + EMNLP/ACL reviewer.

This is a **final re‑audit** of my project on:

> Sparse Autoencoder (SAE)–based multilingual steering in Gemma‑2 models  
> with a focus on Indic languages (Hindi, Urdu, Bengali, Tamil, Telugu).

You have access to previous audit reports and the updated code.

---

## 0. Context and What Changed Since Your Last Audit

Repo root: `sae_multilinugal`

Key files:

- Core code: `config.py`, `data.py`, `model.py`, `evaluation_comprehensive.py`
- Experiments: `experiments/exp1_*.py` … `experiments/exp16_*.py`
- Runners/tools: `run.py`, `smart_run.py`, `summarize_results.py`, `plots.py`
- Manuscript: `paper.tex`, `references.bib`
- Prior audits: `REAUDIT_REPORT.md`, `SKEPTICAL_DEEP_DIVE.md`, `REAUDIT_REPORT_FINAL.md`

Your last report in `REAUDIT_REPORT_FINAL.md` gave a **Conditional Pass** with two critical issues:

1. **Exp2 (steering sanity check)** used FLORES EN sentences both to build steering vectors and as evaluation prompts → train=test reuse.
2. **Exp7 (causal feature probing)** used the same FLORES EN texts for candidate selection and probing prompts → train=test reuse.

You recommended refactoring both to use `data.load_research_data()` so that:

- Steering vectors are built from **train** data (Samanantar + FLORES train, deduped vs test),
- Evaluation prompts come from **held‑out steering prompts** (EVAL_PROMPTS + FLORES EN devtest).

These refactors have now been applied:

- `exp2_steering.py`:
  - Imports `load_research_data`.
  - Uses `data_split.train['en']` and `data_split.train['hi']` for steering vectors.
  - Uses `data_split.steering_prompts` (held‑out prompts) for evaluation.
- `exp7_causal_feature_probing.py`:
  - Imports `load_research_data`.
  - Uses `train['en']` / `train['hi']` for candidate selection.
  - Uses `data_split.steering_prompts` for probing prompts.

All other fixes you previously checked (safe hooks, LaBSE truncation, judge calibration, Exp4/Exp9 using `load_research_data`, etc.) are unchanged.

---

## 1. What I Want You To Re‑Check

### 1.1 Data leakage & train/test separation

Please revisit:

- `experiments/exp2_steering.py`
- `experiments/exp7_causal_feature_probing.py`
- `data.py::load_research_data`

and confirm that:

1. Steering vector construction in Exp2 and Exp7 uses **only** `DataSplit.train` (EN/HI) texts.
2. Evaluation prompts in Exp2 and Exp7 use only `DataSplit.steering_prompts` (EVAL_PROMPTS + FLORES EN devtest).
3. No sentence appears in both the vector‑training set and the evaluation prompts for these experiments.
4. Deduplication logic in `load_research_data` still ensures train vs test FLORES splits are disjoint.

If you find *any* remaining leakage or reuse, please specify:

- `File:line`
- Why it is problematic
- A minimal patch to fix it.

### 1.2 Core infrastructure and main experiments

Please reconfirm (briefly) that:

- `model.py` steering hooks are safe:
  - layer bounds, dimension checks, zero‑norm warnings, dtype/device alignment.
- `evaluation_comprehensive.py`:
  - LaBSE truncation + warning behavior,
  - Jaccard and script/semantic metrics,
  - judge calibration equations (q0, q1, corrected θ̂, CIs).
- Main experiments:
  - **Exp4** (spillover, now using `load_research_data()`),
  - **Exp9** (layer×method×language sweep),
  - **Exp11** (judge calibration),
  - **Exp12** (QA degradation),

are still free of leakage and statistically reasonable given their sample sizes (≈50–100 prompts, ≈100/100 calibration/test, etc.).

### 1.3 Updated overall verdict

Given these latest changes, please provide a **final verdict** along with:

- Whether the previous “Conditional Pass” can now be upgraded to **Pass** for:
  - methodological soundness,
  - statistical adequacy for the *core* story (Exp1/3/4/5/6/9/11/12),
  - and absence of glaring data leakage.
- Any *remaining* minor concerns (e.g., LaBSE limitations, surrogate ground truth for judge calibration, limited 9B scaling) that should be:
  - called out explicitly as caveats in the paper,
  - or framed as future work / exploratory.

If you recommend any further code tweaks or paper caveats, please be specific (file:line for code; section/sentence for paper).

---

## 2. Optional: Quick Checklist for Me

If possible, please finish with a short checklist telling me:

1. Which experiments I should definitely re‑run after your final review (e.g. `exp2`, `exp7`, `exp9`, `exp11`, `exp12`), and in what order.
2. Which experiments/claims you would consider **core, main‑text**, vs **exploratory/appendix**.
3. The top 2–3 residual risks a strong reviewer might still question (even if they are reasonable), so I can pre‑empt them with clear limitations in the paper.

Please be critical but concrete; I want this to be as reviewer‑proof as possible.

