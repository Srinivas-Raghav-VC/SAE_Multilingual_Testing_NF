---
title: "Peeking Inside Gemma: Indic Languages, Sparse Autoencoders, and Steering"
tags: [interpretability, multilingual, gemma, sae, indic]
---

# Peeking Inside Gemma: Indic Languages, Sparse Autoencoders, and Steering

> A walk‑through of the current state of the **Indic SAE + Gemma** project, written for humans rather than just for arXiv.

This note explains:

- what problem we’re trying to solve,
- how the codebase is structured,
- what the first wave of experiments (Exp1–Exp7) are actually telling us,
- how Gemini is used as an LLM‑judge,
- and what’s still marked **N/A** while the long‑running Phase‑2 experiments finish.

It’s intended to be readable in Obsidian (all math is in `$...$` / `$$...$$`).

---

## 1. Why we care: Indic languages inside Gemma

Multilingual LLMs like **Gemma 2** can speak Hindi, Urdu, Bengali, Tamil, Telugu and more.  
But *how* are these languages represented internally?

Some concrete questions:

- Are Hindi and Urdu encoded as “one language, two scripts” or as two separate languages?
- Is there a **language‑agnostic “concept space”** in the middle layers, with language‑specific decoding on top?
- If we “steer” the model towards Hindi, does that also push it towards Urdu and Bengali?
- Can we control language without destroying semantic coherence or QA performance?

To get traction on these, we combine:

- **Gemma 2 models** (2B, later 9B),
- **Gemma Scope sparse autoencoders (SAEs)** on the residual stream,
- **Indic datasets** (Samanantar, FLORES‑200, MLQA, IndicQA),
- and **Gemini** as a calibrated LLM‑judge.

---

## 2. High‑level architecture

At a high level, our pipeline looks like this:

```mermaid
flowchart LR
    A[Input text (EN / HI / UR / ...)] --> B[Gemma 2 (26-layer decoder)]
    B --> C[Residual stream at layer ℓ]
    C --> D[SAE encoder (d=2304 → 16k features)]
    D --> E[SAE decoder (16k → 2304)]
    E --> F[Back into Gemma → logits]
```

In the experiments we attach SAEs at layers:

```text
ℓ ∈ {5, 8, 10, 13, 16, 20, 24}
```

and treat their 16k features as our “basis” for language features, script features, etc.

### 2.1 Conceptual figures

The repo generates quantitative plots under `results/figures/` (pure matplotlib).  
In addition, we use **conceptual diagrams** to tell the story:

- `figures/arch_gemma_sae_pipeline.png`  
  High‑level view of Gemma + SAEs + datasets.

- `figures/concept_language_space.png`  
  Early / mid / late language‑space schematic:
  - early: blobs for each language, mostly overlapping (orthography + tokenization),
  - mid: shared concept space (semantic alignment),
  - late: separated language ellipses (language‑specific decoding).

- `figures/concept_detectors_generators.png`  
  Detectors vs generators across depth:
  - detector counts (monolinguality) rising towards late layers,
  - generator “sweet spot” for steering in mid‑late layers.

You can drop these into the repo’s `figures/` directory and refer to them from the paper and from this note:

```markdown
![Architecture overview](figures/arch_gemma_sae_pipeline.png)

![Language space across layers](figures/concept_language_space.png)

![Detectors vs generators](figures/concept_detectors_generators.png)
```

---

## 3. Core technical ingredients

### 3.1 Sparse autoencoders on the residual stream

At layer $\ell$ we take Gemma’s residual stream activations

$$
h_{\ell, t} \in \mathbb{R}^{d}, \quad d = 2304
$$

and feed them into a sparse autoencoder:

$$
z_{\ell,t} = \mathrm{Enc}_\ell(h_{\ell,t}) \in \mathbb{R}^{m}, \quad m = 16384
$$

$$
\hat h_{\ell,t} = \mathrm{Dec}_\ell(z_{\ell,t}) \approx h_{\ell,t}
$$

In the Gemma Scope SAEs:

- $z_{\ell,t}$ is **sparse** (most coordinates $= 0$),
- each feature $j$ has an associated decoder direction $w_{\ell,j} \in \mathbb{R}^d$ (a column of $W_{\text{dec}}$),
- we treat each feature as a candidate “concept” (language, script, grammar, semantics, …).

We work with **binary activations**:

$$
a_{\ell,j}(x, t) = \mathbf{1}[z_{\ell,t,j} > 0]
$$

and **activation rates** per language $L$:

$$
\hat P_\ell(j \mid L)
  = \frac{\sum_{x \in \mathcal{D}_L}\sum_t a_{\ell,j}(x,t)}
         {\sum_{x \in \mathcal{D}_L} |x|}
$$

### 3.2 Monolinguality: “detector” score

To find *language detectors* for a target language $T$ (e.g. Hindi), we use a **monolinguality score**:

$$
M_{\ell,j}(T)
  = \frac{\hat P_\ell(j \mid T)}
         {\max_{L \neq T} \hat P_\ell(j \mid L)}
$$

If $M_{\ell,j}(T) > 3$, feature $j$ is at least **3× more active** for $T$ than for any other language, and we call it a **$T$‑selective detector** at layer $\ell$.

These detectors are great for *recognizing* that text is Hindi—but, as we’ll see, they’re terrible at *causing* Hindi generation when used as steering directions.

### 3.3 Jaccard overlap: shared features across languages

For language $L$ at layer $\ell$, define the set of “active” features

$$
\mathcal{F}_\ell(L) = \{ j : \hat P_\ell(j \mid L) > \tau \}, \quad \tau \approx 0.01.
$$

To measure shared feature sets between languages $L_1$ and $L_2$, we use **Jaccard overlap**:

$$
J_\ell(L_1, L_2)
  = \frac{|\mathcal{F}_\ell(L_1) \cap \mathcal{F}_\ell(L_2)|}
         {|\mathcal{F}_\ell(L_1) \cup \mathcal{F}_\ell(L_2)|}
  \in [0, 1].
$$

This is how we quantify:

- how similar Hindi–Urdu features are,
- how that compares to Hindi–English,
- and how much of the HI–UR union is “script‑specific” vs “semantic”.

### 3.4 Steering in hidden space

For steering we add a vector $v_\ell \in \mathbb{R}^d$ to the residual stream at a layer:

$$
h'_{\ell,t} = h_{\ell,t} + \alpha v_\ell
$$

for a strength $\alpha$ (e.g. $0.5 \ldots 4.0$).  
We construct $v_\ell$ in several ways:

1. **Dense activation difference** (no SAE):
   $$
   v_\ell^{\text{dense}}(T \leftarrow S)
     = \mathbb{E}[\bar h_\ell \mid T] - \mathbb{E}[\bar h_\ell \mid S]
   $$
   where $\bar h_\ell$ is the mean residual over tokens.

2. **SAE activation‑difference**:
   - pick top‑$k$ features by $\hat P_\ell(j \mid T) - \hat P_\ell(j \mid S)$,
   - average their decoder directions $w_{\ell,j}$.

3. **Monolinguality‑based**:
   - pick top‑$k$ features by $M_{\ell,j}(T)$,
   - average their decoder directions.

4. **Attribution‑based** (Exp10):
   - ablate one feature at a time and see how much it hurts Hindi success,
   - build a steering vector from features with the largest causal effect.

---

## 4. Datasets and evaluation

We try to keep the dataset story clean:

- **Samanantar (AI4Bharat)** – large EN–Indic parallel corpus, used for *training* steering directions and feature statistics for HI/BN/TA/TE.
- **FLORES‑200** – small, high‑quality parallel corpus, used for:
  - cross‑lingual overlap experiments (HI–UR–EN etc.),
  - FLORES‑only languages (UR/DE/AR),
  - some steering sanity checks.
- **MLQA / IndicQA** – QA datasets used in later experiments (Exp12) to measure how steering affects task performance.

For evaluation we combine:

- **Script ratios** from Unicode ranges (Latin, Devanagari, Arabic, Bengali, Tamil, Telugu, …) with a dominance margin to reject code‑mix.
- **Semantic similarity** via LaBSE:
  $$\text{sim}(x,y) = \cos(e_{\text{LaBSE}}(x), e_{\text{LaBSE}}(y)),$$
  with a threshold (e.g. $0.7$) for “semantics preserved”.
- **Degradation** via $n$‑gram repetition rates ($n=3,5$).
- **LLM‑as‑judge (Gemini)** with calibration à la “no free labels”:
  - raw Gemini scores for language, faithfulness, coherence,
  - bias‑corrected accuracy estimates using a small human‑labeled calibration set.

---

## 5. What the first experiments show

The codebase currently has 16 experiments (`exp1`–`exp16`).  
Here we summarize the ones that have produced stable results so far (Exp1–Exp7) plus the design of the later ones.

### 5.1 Exp1 – Language detectors (monolinguality)

**Question.** Do we actually find *language‑selective* features in the SAEs?

**Method.**

- For each layer $\ell$ and language $L \in \{\text{EN, HI, UR, BN, TA, TE, DE, AR}\}$:
  - compute $M_{\ell,j}(L)$ for all 16k features,
  - count features with $M_{\ell,j}(L) > 3$.

**What we saw.**

- For every language, there are **hundreds** of strongly selective features per layer.
- The **total number of language‑selective features increases with depth**, peaking at the last SAE layer (24).  
  In one run, total “any‑language selective” counts across all languages were roughly:

  ```text
  Layer:   5   8   10   13   16   20   24
  Count: 797 764 884 777 1084 1506 1918
  ```

**Take‑away.**

- The **“detector” features (high monolinguality)** are very much real.
- They are not maximal in the “mid layers”; instead they **concentrate towards late layers**, close to the logits.
  - This falsifies a naïve “mid‑layer peak of language identity” hypothesis.
  - It suggests: mid layers are more about shared semantics, while late layers carve out language identity for decoding.

### 5.2 Exp2 – Steering method comparison (EN→HI)

**Question.** Which steering method and layer actually make the model *generate* Hindi from English prompts?

**Setup.**

- Source: EN, Target: HI.
- Layers tested: 5, 13, 20, 24.
- Methods: `dense`, `activation_diff`, `monolinguality`, `random`.
- Strength grid: $\alpha \in \{0.5, 1.0, 2.0, 4.0\}$.
- Success criterion: output dominated by Devanagari script + not obviously degraded.

**Key pattern.**

- **Dense & activation‑diff SAE**:
  - Layers 5, 13, 20: script success can reach **90–100%** at moderate strengths.
  - Layer 24: **0% success for all strengths**.
- **Monolinguality & random**:
  - Essentially **0% success at all layers**.

**Interpretation.**

- High‑$M$ features are **detectors**, not generators:
  - they light up on Hindi text,
  - but steering along them doesn’t drive the model *into* Hindi.
- There is a **“generator window”** in the **mid–late layers (≈13–20)** where steering directions derived from activation differences actually work.
- The **final layer (24)** seems “too late”: the geometry of the residual stream there is such that simple additive steering no longer effectively changes language.

This is the core “detectors vs generators” story:

- detectors accumulate late,
- generators live slightly earlier.

### 5.3 Exp3 – Hindi–Urdu overlap and script vs semantics

**Question.** Are Hindi and Urdu “one language, two scripts” in the SAE space?

**Setup.**

- Use parallel FLORES‑200 sentences for EN, HI (Devanagari), UR (Arabic).
- For each layer $\ell$:
  - compute active feature sets $\mathcal{F}_\ell(\text{HI})$, $\mathcal{F}_\ell(\text{UR})$, $\mathcal{F}_\ell(\text{EN})$,
  - compute Jaccard overlaps:
    - $J_\ell(\text{HI}, \text{UR})$,
    - $J_\ell(\text{HI}, \text{EN})$,
  - classify features as:
    - **semantic**: active in HI, UR, and EN,
    - **script‑specific**: active in exactly one of HI/UR.

**What we saw (first run).**

- For all layers,
  $$J_\ell(\text{HI}, \text{UR}) \gg J_\ell(\text{HI}, \text{EN})$$
  with HI–UR Jaccard in the **0.93–0.99** range.
- The fraction of **script‑specific features** in the HI–UR union is:
  - typically between **1–3%**,
  - rising to about **6–7%** at a couple of layers (10 and 24),
  - never close to 50%.
- In other words, **90–99% of HI–UR features look semantic/shared**, not purely script‑driven.

**Take‑away.**

- The SAE features support the intuition that **Hindi and Urdu share a huge semantic backbone**, with a *thin shell* of script‑specific detectors for Devanagari vs Arabic.
- This supports treating HI–UR as essentially “one language, two scripts” in representational space.

### 5.4 Exp5 & Exp6 – Hierarchical structure and script controls

**Goal.** Understand how language information flows across layers and separate script from semantics more carefully.

#### Exp5 – Hierarchical feature analysis

We compute, for each layer:

- **shared features**: active for all languages (EN, HI, UR, BN, TA, DE),
- **Indic‑only features**: active for Indic but not for EN/DE/AR,
- per‑language feature counts,
- HI–UR overlaps across early / mid / late bands.

Qualitative pattern:

- **Early layers (5, 8)**:
  - High shared feature counts,
  - very high HI–UR overlap (≈98%),
  - consistent with tokenization + low‑level orthography.

- **Mid layers (10, 13, 16)**:
  - Slight drop in shared counts but still large,
  - HI–UR overlap still very high (≈96–97%),
  - Indic‑only features begin to separate from EN/DE.

- **Late layers (20, 24)**:
  - Shared counts comparable or slightly lower,
  - HI–UR overlap still >90% but a bit lower than mid,
  - more room for language‑specific decoding and steering.

#### Exp6 – Script vs semantics controls (HI transliteration + noise)

To disentangle script vs language:

- We compare activations for:
  - Hindi in Devanagari,
  - the **same Hindi sentences transliterated into Latin script**,
  - **Devanagari noise** (random Hindi‑looking characters),
  - Urdu and Bengali.

Qualitative findings:

- HI(Deva) vs HI(Latin) still share a **large majority of features** at all layers.
- The number of **“script‑only” Devanagari features** (active on HI(Deva) but not HI(Latin)) is modest—hundreds out of thousands, rarely above ≈5–6% of the union.
- Features that are active on HI, UR, BN simultaneously look like good candidates for **family‑level semantic features**.

**Take‑away.**

- Script sensitivity is **real but thin**: most features that matter for content appear robust across scripts and closely related languages.
- This strengthens the interpretation of Exp3’s “semantic features” as genuinely content‑driven rather than an artifact of script.

### 5.5 Exp7 – Causal single‑feature probing (design + status)

**Goal.** Move from correlation to causation for individual features.

- For a mid/late layer (currently 5, 13, 20, 24 are used), we:
  - select candidate features via activation‑difference and monolinguality,
  - for each feature:
    - run baseline generations,
    - inject a steering vector that only uses that feature’s decoder direction,
    - measure change in Hindi script success and degradation.

Logs confirm:

- occlusion and single‑feature steering run end‑to‑end and produce JSON summaries in `results/exp7_causal_feature_probing.json`.

Full statistical analysis of those distributions is still ongoing, so we keep detailed numbers as **N/A** for now, but the infrastructure is in place.

---

## 6. Later experiments (Exp8–Exp16): what they’re meant to answer

The remaining experiments are where the story becomes fully “paper‑level”. They are implemented and running, but not all have stabilized yet, so we treat detailed metrics as **pending**.

- **Exp8 – Scaling to 9B and low‑resource languages**
  - Compare 2B vs 9B feature counts and EN→HI steering effectiveness.
  - Include low‑resource Indic (as, or) and non‑Indic (vi, ar).

- **Exp9 – Full layer × method × language sweep (with Gemini judge)**
  - For each target language (HI, UR, BN, TA, TE, DE, AR):
    - sweep layers 5–24, methods {dense, activation_diff, monolinguality, random}, strengths, prompts.
    - measure script success, semantics, degradation, and Gemini‑based quality.
  - This gives the **complete “steering landscape”**.

- **Exp10 – Attribution‑based steering (occlusion SAEs)**
  - Use occlusion to score features by causal importance, build steering vectors from top‑k.
  - Compare to dense and activation‑diff steering.

- **Exp11 – Calibrated Gemini judge**
  - Calibrate Gemini’s language/faithfulness/coherence scores against a small human‑labeled set.
  - Compute bias‑corrected accuracy and confidence intervals.

- **Exp12 – QA degradation under steering**
  - Evaluate MLQA/IndicQA with baseline vs steering, to quantify task‑level trade‑offs.

- **Exp13 – Group ablation of script vs semantic feature sets**
  - Ablate script‑specific vs script‑invariant feature groups and see how each affects script vs semantic metrics.

- **Exp14 – Language‑agnostic mid‑layer space**
  - Measure EN↔Indic cosine similarities across layers (using pooled hidden states).
  - Look for a peak in cross‑lingual alignment in mid layers.

- **Exp15 – Directional symmetry (EN→Indic vs Indic→EN)**
  - Compare how easy it is to steer EN→L vs L→EN in the same layer.

- **Exp16 – Code‑mix robustness**
  - Test EN→HI steering on clean English, EN+Devanagari, and Hinglish‑style prompts.

Once those results are solid, the “blog version” of this note can be updated with concrete numbers; for now they’re part of the **experimental design**, not the finalized findings.

---

## 7. Running the code yourself

If you have an A100‑class GPU and the repo checked out:

```bash
cd SAE_Multilingual_Testing_NF   # or sae_multilinugal in your path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN="hf_..."              # your Hugging Face token
export GOOGLE_API_KEY="AIza..."       # Gemini API key (used for judge + labels)
```

Sanity check:

```bash
python run.py --validate
```

Run everything (this is long, best left overnight):

```bash
python run.py --all 2>&1 | tee logs/run_all_$(date +%F_%H-%M).log
```

Or just the Phase‑2 experiments:

```bash
python run.py --exp8 --exp9 --exp10 --exp14 --exp15 --exp16 \
  2>&1 | tee logs/run_phase2_$(date +%F_%H-%M).log
```

Plots:

```bash
python plots.py
```

will populate `results/figures/` with all the quantitative figures used in the paper (feature counts, steering heatmaps, overlap curves, etc.).

---

## 8. Big picture so far

Even with only Exp1–Exp7 fully digested, a coherent story is starting to emerge:

1. **Detectors vs generators are real and separate.**
   - High‑monolinguality features are *detectors* of languages and scripts.
   - Useful steering directions are *generator*‑like vectors (dense or activation‑diff) living in a mid–late window.

2. **Language identity peaks late, not in the middle.**
   - The count of strongly language‑selective features rises towards layer 24.
   - This suggests Gemma resolves “which language?” very close to the output.

3. **Hindi–Urdu behave as “one language, two scripts”.**
   - Jaccard overlaps $J_\ell(\text{HI}, \text{UR})$ are ≈0.93–0.99 across all SAE layers.
   - Script‑specific features form a small shell around a large shared semantic core.

4. **Script is a thin layer on top of semantics.**
   - Transliteration and Devanagari noise experiments show that most important features are script‑robust and/or shared across related Indic languages.

5. **Steering is powerful but delicate.**
   - In the right layer window, EN→HI steering can reach very high script success.
   - Detector‑based and final‑layer steering fail, teaching us what **doesn’t** work.

The remaining experiments (scaling, spillover, symmetry, QA degradation, code‑mix robustness, calibrated judge) will turn this into a full empirical story—and a proper preprint.  
For now, this note is a snapshot of what we’ve already *seen* and how the different pieces fit together.

--- 

*Author: Srinivas Raghav V C (IIIT Kottayam)*  
*Project: SAE Multilingual Steering in Gemma*

