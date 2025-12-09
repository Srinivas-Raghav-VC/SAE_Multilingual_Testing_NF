# Full Re‑Audit Prompt for Gemini 3‑Pro

You are a **very skeptical senior ML + NLP researcher** with strong experience in:

- Mechanistic interpretability (esp. SAEs, superposition)
- Multilingual LLMs (Gemma, LLaMA, etc.)
- Evaluation methodology (LLM‑as‑judge, QA, statistics)

Please behave like a tough advisor + EMNLP/ACL reviewer.

I’m working on a project on **Sparse Autoencoder (SAE)–based multilingual steering in Gemma models**, focused on Indic languages (HI, UR, BN, TA, TE). I want you to critically re‑evaluate the *entire* project: code, experiments, statistics, data usage, paper, and figures.

You have access to the full repo and previous internal audits.  
Assume I want *zero surprises* when I send this as my first preprint.

---

## 0. Repository & Context

Repo root: `sae_multilinugal`

Key files:

- **Core code**
  - `config.py`
  - `data.py`
  - `model.py`
  - `evaluation_comprehensive.py`
- **Experiments**
  - `experiments/exp1_*.py` … `experiments/exp16_*.py`
- **Runners / utilities**
  - `run.py`
  - `smart_run.py`
  - `summarize_results.py`
  - `plots.py`
  - `test_gemini_api.py`
- **Manuscript**
  - `paper.tex`
  - `references.bib`
- **Audits / notes**
  - `HYPOTHESIS_EVALUATION.md`
  - `LITERATURE_ALIGNMENT_CHECK.md`
  - `REAUDIT_REPORT.md`  (Gemini‑3‑Pro “Ruthless Re‑Audit”)
  - `SKEPTICAL_DEEP_DIVE.md` (another skeptical pass)
- **Results (after runs)**
  - `results/*.json`
  - `results/summary_report.txt`
  - `results/figures/*`

Please *read* `REAUDIT_REPORT.md` and `SKEPTICAL_DEEP_DIVE.md` for context, but **do not trust them automatically**; instead, use them as hints and re‑check things yourself.

---

## 1. What the Project Is Trying to Do

High‑level goal:

- Use **Gemma‑2** models (2B main, 9B as a sanity check) with **Gemma Scope SAEs** to:
  - Characterize how Indic languages (Hindi, Urdu, Bengali, Tamil, Telugu) are represented across layers.
  - Distinguish **detector** features (high monolinguality) from **generator** directions (effective steering).
  - Study **cross‑lingual geometry** (HI–UR vs HI–EN overlap, script vs semantics).
  - Evaluate **steering methods** (dense, activation‑diff SAE, monolinguality SAE, attribution‑based).
  - Measure **spillover** (steering one language affecting others).
  - Quantify **QA degradation** under steering with a **calibrated Gemini judge**.

Model & SAEs:

- Model: Gemma‑2 2B (main) and a lighter analysis on Gemma‑2 9B.
- SAEs: Gemma Scope residual‑stream SAEs (16k JumpReLU) attached at layers:
  - `[5, 8, 10, 13, 16, 20, 24]`

Experiments and intended roles:

- **Geometry / representations**
  - **Exp1** – Feature discovery:
    - Compute activation rates & monolinguality scores M(feature, language).
    - Count strongly language‑selective (M>3) features per layer and language.
  - **Exp3** – Hindi–Urdu overlap:
    - Active feature sets per language & layer, Jaccard(HI, UR) vs Jaccard(HI, EN).
    - Script vs semantic feature split (HI & UR but not EN vs HI & UR & EN).
  - **Exp5** – Hierarchical analysis:
    - Feature counts per language/layer; which features persist or emerge.
  - **Exp6** – Script vs semantics controls:
    - Use transliteration and Devanagari noise to operationalize “script‑only” vs “script‑robust/semantic” features.
- **Steering & layers**
  - **Exp2** – Steering sanity check:
    - Dense vs activation‑diff SAE vs monolinguality SAE vs random at a few layers.
    - Uses FLORES EN prompts to check that activation‑diff & dense steer, monolinguality/random don’t.
  - **Exp9** – Core layer × method × language sweep:
    - For each target language (HI, UR, BN, TA, TE, DE, AR), test steering at layers `[5,8,10,13,16,20,24]`, methods:
      - Dense, activation‑diff SAE, monolinguality SAE, random.
    - Evaluate script success, LaBSE similarity, degradation, and calibrated judge scores.
  - **Exp10** – Attribution‑based steering:
    - Use occlusion/attribution to identify causally important SAE features and build steering vectors from them.
- **Spillover / clustering**
  - **Exp4** – Spillover analysis:
    - **EN→HI** steering: measure distribution over HI/UR/BN/TA/TE/DE/AR.
    - **EN→DE** control: show that non‑Indic steering does *not* spuriously increase Indic outputs.
    - Earlier versions used the same FLORES EN sentences for both steering vector estimation and prompts; this has now been refactored to use `load_research_data()` so train sentences and prompts are separated.
- **Causality in SAE space**
  - **Exp7** – Single‑feature causal probing:
    - For selected features at certain layers, steer using a single SAE feature vector and measure Δscript, Δsemantics, Δdegradation.
    - Now uses ≥50 FLORES EN prompts (was previously under‑powered).
  - **Exp13** – Group ablation:
    - Ablate “script‑specific” vs “script‑robust/semantic” feature groups and measure effects.
- **Scaling**
  - **Exp8** – Gemma‑2‑9B scaling:
    - Limited analysis: feature counts, a small amount of steering; meant as a **sanity check**, not a full replication.
- **Judge & QA**
  - **Exp11** – Judge calibration:
    - Gemini as judge, calibrated against structural metrics (script + LaBSE).
    - Lee‑style bias correction: estimate q0, q1; compute corrected θ̂ + CIs.
  - **Exp12** – QA degradation under steering:
    - Use MLQA & IndicQA QAs; evaluate baseline vs steered QA performance and degradation, plus calibrated judge scores.
- **Other exploratory**
  - **Exp14** – Cross‑lingual alignment:
    - Compare sentence embeddings across languages across layers (EN, HI, UR, BN, TA, TE).
  - **Exp15** – Directional symmetry:
    - EN→Indic vs Indic→EN steering patterns.
  - **Exp16** – Code‑mix robustness:
    - Hinglish / mixed‑script prompts, synthetic for now.

---

## 2. Recent Patches (Relative to Earlier Audits)

Based on `REAUDIT_REPORT.md`, `SKEPTICAL_DEEP_DIVE.md`, and subsequent work, we have already patched:

- **Steering hooks (`model.py`)**
  - Check that the model is loaded.
  - Assert `0 ≤ layer < len(self.model.model.layers)`.
  - Assert steering vector dimension matches `config.hidden_size`.
  - Move steering vector to the correct device and match hidden dtype (e.g., bfloat16) inside the hook.
  - Check for near‑zero norm steering vectors and warn.

- **Activation rates & monolinguality**
  - Exp1 and Exp3 now use consistent definitions:
    - Activation rate: `(acts > 0).sum(dim=0) / total_tokens`.
    - Monolinguality: M = P(feature | target) / max_{other} P(feature | other), with:
      - eps floor on denominator,
      - tiny P(target) → M set to 0.

- **Jaccard & script/semantic ratios**
  - Correct Jaccard: |A∩B| / |A∪B|, with assertion 0 ≤ J ≤ 1.
  - Empty union handled explicitly.
  - Script vs semantic fractions now use the union set; script‑specific never > 100%.

- **LaBSE**
  - `_truncate_for_semantic_model` truncates to 512 tokens (approx via whitespace split).
  - Warns only once per process via `_TRUNCATION_WARNED` to avoid log spam.
  - If LaBSE model can’t load, similarity returns −1.0 and the rest of the pipeline handles that.

- **Sample sizes**
  - Exp2, Exp4, Exp7, Exp9 now use ≥50–200 prompts (often FLORES EN plus EVAL_PROMPTS) via:
    - Local logic (Exp2, Exp7), and
    - Global logic in `data.py::load_research_data` for steering prompts.
  - Exp11 uses ≈100 calibration + ≈100 test examples per language.

- **Exp4 hygiene (train ≠ test sentences)**
  - Refactored `exp4_spillover.py` to use:
    - `load_research_data()` → get `train` and `steering_prompts`.
    - Steering vectors built from `train` (Samanantar + FLORES train).
    - Prompts from `DataSplit.steering_prompts` (EVAL_PROMPTS + held‑out FLORES EN devtest).
  - This eliminates using the exact same FLORES EN sentences to both build the steering vector and evaluate spillover.

- **Exp7 under‑powered prompt set**
  - `exp7_causal_feature_probing.py` now:
    - Uses FLORES EN texts for prompts (`texts_en`), not just `EVAL_PROMPTS`.
    - Ensures `num_prompts = min(max(N_SAMPLES_EVAL, 50), len(texts_en))`, with a warning if <20.

- **Judge calibration**
  - Exp11 implements Lee‑style calibration and logs/writes:
    - `n_calib_0`, `n_calib_1`, `n_test` per language.
  - Exp9/10/12 read the calibration table, distinguish raw vs corrected judge scores, and log relevant counts.

- **Data splits & steering prompts**
  - `data.py::load_research_data`:
    - Dedupes train vs test (removes any test sentence from train).
    - Builds a `steering_prompts` list: starts from `EVAL_PROMPTS`, then augments with FLORES EN devtest up to `N_STEERING_EVAL`.
    - Returns a `DataSplit` with `train`, `test`, `steering_prompts`, and `qa_eval`.

You should check that all of this is *actually* correct in the current repo, but this is the intended state.

---

## 3. Code & Experiment Audit

Please go through:

- `model.py`
- `data.py`
- `evaluation_comprehensive.py`
- All `experiments/exp*.py`

and:

### 3.1 Correctness & Silent‑Failure Modes

1. **Steering correctness**
   - Any remaining ways to:
     - Create zero or wrong‑shaped steering vectors without warning?
     - Hook the wrong layer if Gemma’s interface changes?
     - Have dtype/device mismatches that could silently no‑op or distort steering?
   - Verify **LaBSE**:
     - 512‑token handling,
     - Behavior if the model is missing,
     - Any subtle bias/truncation issues for QA contexts.
   - Judge calibration usage:
     - What happens if calibration JSON is missing or partial?
     - Any path where **raw** judge scores are mis‑treated as calibrated?

   For each issue, give:
   - `File:line`,
   - Why it matters (bug vs fragile assumption vs noise),
   - A minimal patch suggestion.

### 3.2 Statistical Adequacy (Per Experiment)

For **every experiment** (Exp1–Exp16):

- Check actual sample sizes in code:
  - Exp1/3/5/6 – #sentences per language,
  - Exp2/4/7/9 – #prompts,
  - Exp11/12 – #calibration and #test examples,
  - Exp8/14/15/16 – #examples.
- Compare against effect sizes:
  - 0% vs ≫50% steering success,
  - Jaccard 0.85 vs 0.95,
  - QA drops of ~10–20 points.

For each experiment, classify:

- **Core & statistically solid**
- **Acceptable but exploratory**
- **Underpowered / not suitable for strong claims**

and suggest concrete changes (e.g., increase `N_SAMPLES_EVAL`, or move to appendix).

### 3.3 Data Reuse, Leakage, Circularity

- Verify train vs test splits in `data.py` (Samanantar vs FLORES train vs devtest, MLQA, IndicQA) and that dedup works as intended.
- Look carefully at **FLORES EN reuse**:
  - Are there still any places where the *same* EN sentences are used both to build steering vectors and to evaluate them?
- Check for subtle circularity:
  - Script + LaBSE metrics define success; the same are used as “ground truth” for judge calibration.
  - Any path where prompts used to configure steering are reused in evaluation such that it looks like “training on test”?

Classify each concern as **major**, **moderate**, or **minor** and recommend either:

- A code fix, or
- A specific caveat in the paper.

### 3.4 Judge Calibration Implementation & Usage

- Verify in Exp11 and `evaluation_comprehensive.py` that:
  - q0, q1 are computed correctly,
  - Bias‑corrected θ̂ and CIs match Lee‑style formulas,
  - Edge cases (judge worse than random) are handled reasonably.
- Check that Exp9/10/12:
  - Load calibration correctly,
  - Use corrected scores when they say “calibrated judge”,
  - Log/serialize `n_calib_0`, `n_calib_1`, `n_test`.

For any mis‑calculation or misuse, give exact locations and patches.

---

## 4. Paper + Figures vs Code/Results

Treat `paper.tex` + `references.bib` as a submission where you also see the code and results.

### 4.1 Claims vs Evidence

For these central claims, say **Strongly supported / Supported but needs softening / Not supported**:

1. **Detector vs generator across depth**
   - SAE monolinguality (M>3) “detector” features peak in late layers.
   - Effective steering (dense + activation‑diff) happens mainly in mid–late “generator window” (≈10–20).
   - Final layer (24) is ineffective for steering.

2. **Detector vs generator dissociation**
   - Monolinguality‑selected features detect language well but yield ≈0% steering success.
   - Activation‑diff and dense vectors work; random features don’t.

3. **Hindi–Urdu vs Hindi–English geometry**
   - Jaccard(HI,UR) > 0.93 across layers and always > Jaccard(HI,EN).
   - Script‑specific features <10% of HI–UR union across layers.

4. **Spillover / Indic clustering**
   - EN→HI steering increases UR/Indic outputs more than DE/AR.
   - EN→DE control shows little increase in Indic outputs.

5. **QA degradation & calibrated judge**
   - Steering strengthens target‑script dominance but increases repetition and reduces QA accuracy.
   - Calibrated judge scores meaningfully quantify this.

For each claim:

- Check code + typical results (JSON, plots).
- If needed, propose revised wording (LaTeX snippets) with correct caveats (“in our SAE decomposition”, “for Gemma‑2 2B”, “exploratory”, etc.).

### 4.2 Literature Alignment

Check `references.bib` & citations for:

- Activation Addition / representation engineering (Turner et al., RepE‑style work),
- Anthropic SAE / monosemanticity / superposition (Templeton et al., GemmaScope, SAE‑Lens),
- Multilingual representation & “English latent language” (Wendler et al. and similar),
- Indic resources: Samanantar, IndicNLP Suite, MuRIL, FLORES‑200, MLQA, IndicQA,
- LLM‑as‑judge & bias/correction (Lee‑style).

If anything important is missing/mis‑cited:

- Tell me which references to add and
- Exactly where in `paper.tex` to cite them.

### 4.3 Edge Cases, Limitations, Caveats

Check whether the paper explicitly acknowledges:

- FLORES EN reuse (vector estimation vs prompts; cross‑modality but same corpus),
- New vs old findings:
  - SAE‑based detector peak vs neuron‑level mid‑layer selectivity; framed as a property of the **SAE view of Gemma**,
- Scope:
  - Exp7, 15, 16 are exploratory/appendix,
  - Exp8 (9B) is a limited sanity check,
- Judge calibration:
  - Uses script+LaBSE as surrogate ground truth,
  - Calibration sample sizes are moderate, not huge.

If anything is missing or weak, propose concrete LaTeX edits.

### 4.4 Figures & Visual Story

Assume quantitative plots from `plots.py` exist:

- Detector counts vs layer (Exp1),
- HI–UR vs HI–EN Jaccard (Exp3),
- Steering heatmaps (layer × method × language; Exp9),
- QA degradation curves (Exp12),
- Calibrated judge accuracy bars (Exp11).

And conceptual diagrams from an image model:

- Architecture overview (Gemma+SAE+datasets),
- Detectors vs generators,
- Early/Mid/Late concept space,
- Spillover sketch.

Please:

- Check **consistency**:
  - Does layer 24 actually look ineffective in steering heatmaps?
  - Do conceptual diagrams match quantitative patterns?
- Check **labelling**:
  - Conceptual figures clearly marked as such (“(conceptual)”).

Recommend:

- Which 4–6 figures belong in the main paper vs appendix,
- Caption changes for precise, reviewer‑proof wording,
- Any additional schematic that would clarify the story.

---

## 5. Specific Doubts to Judge (Major / Moderate / Minor)

Please explicitly assess each as **major / moderate / minor** issue, how likely a strong reviewer is to attack it, and the best mitigation (extra runs, caveats, move to appendix, etc.):

1. **SAE vs neuron‑level interpretations**
   - Late detector peak is seen in SAE feature space; neuron‑level work finds mid‑layer selectivity.
   - Is “SAE decomposition of Gemma’s residual stream reveals…” an honest framing?

2. **Correlation vs causation in SAE space**
   - Monolinguality and activation‑diff are correlational.
   - Exp7/10/13 add causal evidence.
   - Are these enough for a causal detector vs generator story, or should we frame more cautiously?

3. **Script vs semantics split**
   - Defined via HI/UR/EN activation patterns + transliteration + noise + group ablation.
   - Is this robust enough to call features “script‑sensitive” vs “script‑robust/semantic”?

4. **FLORES EN reuse**
   - Remaining cross‑modality reuse (vector estimation vs prompts).
   - Is this acceptable for a first paper, or do we need stronger caveats / further splits?

5. **LaBSE semantics for Indic**
   - Is relying on LaBSE for EN+Indic semantics reasonable, or do we need explicit discussion of its limitations and potential bias?

6. **Judge calibration assumptions**
   - Calibration uses structural metrics as surrogate ground truth; no human labels.
   - Is it still fair to call this a “calibrated judge” prominently?

7. **9B scaling story (Exp8)**
   - Evidence is light; mostly feature counts and a bit of steering.
   - Should scaling claims be strictly limited to one cautious sentence?

8. **Code‑mix robustness (Exp16)**
   - Synthetic Hinglish/noise only.
   - Should this be purely appendix‑level with very cautious language?

9. **Per‑feature causal effects (Exp7)**
   - Still somewhat noisy; no modeling of feature interactions.
   - Should any main‑text claims rely on Exp7, or is it supporting evidence only?

---

## 6. Final Checklist I Need From You

Please finish with a concise checklist that tells me:

1. **Experiments to re‑run**  
   After any patches you recommend, which experiments must be re‑run, with explicit commands, e.g.:
   ```bash
   python run.py --exp2 --exp4 --exp7 --exp9 --exp11 --exp12
   python summarize_results.py
   python plots.py
   ```
   (Modify this as you see fit.)

2. **Core vs exploratory**  
   - Which experiments/claims are **core, high‑confidence** for main‑text claims.
   - Which ones must be clearly labelled **exploratory** and/or moved to appendix.

3. **Top remaining risks**  
   - The 3–5 most likely attack surfaces for a strong reviewer/advisor,
   - How you recommend mitigating each (more data, extra runs, stronger caveats, or future‑work framing).

Please be as concrete and critical as possible: assume I want to close every obvious attack surface before posting this as my first preprint.

