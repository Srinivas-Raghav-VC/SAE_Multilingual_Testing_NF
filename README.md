# SAE Multilingual Steering (Gemma + Gemma Scope SAEs)

Research codebase for **measuring and steering language behavior** in Gemma models using **Sparse Autoencoders (SAEs)**.

## What This Repo Does

- Loads Gemma 2 (2B; optional 9B sanity check) and Gemma Scope residual-stream SAEs.
- Runs a **core, paper-oriented** set of experiments:
  - Language-selective feature discovery (monolinguality detectors).
  - Cross-lingual overlap (Hindi–Urdu vs Hindi–English Jaccard).
  - Steering sweeps (layer × method × language × strength).
  - Spillover + **EN→DE control**.
  - QA degradation under steering (MLQA + Indic QA).
  - Causal checks (occlusion attribution; group ablations).
- Writes results to `results/*.json`, and figures to `results/figures/` via `plots.py`.

## Setup

```bash
# Hugging Face (optional but recommended)
export HF_TOKEN=hf_...

# Gemini judge (optional; used when enabled)
export GEMINI_API_KEY=...
# or: export GOOGLE_API_KEY=...

pip install -r requirements.txt
```

## Run

```bash
# Sanity/rigor checks
python run.py --validate

# Core suite (recommended)
python run.py --all

# Scaling runs on Gemma-2-9B (+ 9B SAEs)
# Writes separate *_9b.json files under results/
python run.py --all --use_9b

# Or run specific experiments
python run.py --exp1   # feature discovery
python run.py --exp3   # Hindi–Urdu overlap
python run.py --exp9   # main steering sweep (can be expensive)
```

### Smart runner (waits for GPU)

```bash
python smart_run.py
```

## Gemini rate limits / cost control

Exp9 can generate many judge calls. To subsample:
```bash
export GEMINI_SAMPLE_RATE=0.2   # judge ~1/5 prompts
export GEMINI_MAX_RPM=60        # throttle (per-process)
export GEMINI_MAX_RETRIES=5     # retry/backoff on 429s
```

## Steering schedules (optional)

Steering is implemented as activation addition, with optional schedules:
```bash
export STEERING_SCHEDULE=constant        # default
export STEERING_SCHEDULE=generation_only # CAA-style (after the prompt)
export STEERING_SCHEDULE=exp_decay
export STEERING_DECAY=0.9
```

## Better language ID for spillover controls (optional, recommended)

Exp4 spillover includes Latin-script controls (EN/DE/ES/FR) and Arabic-script
controls (AR vs UR). Script-based detection alone is insufficient for these.

We support an optional fastText-based LID backend via `fast-langdetect`:

```bash
pip install fast-langdetect
python scripts/download_fasttext_lid.py          # downloads lid.176.ftz to models/

export LID_BACKEND=fasttext
export FASTTEXT_LID_MODEL_PATH=models/lid.176.ftz
```

If you cannot install `fast-langdetect`, you can still run with the default
regex heuristics (less reliable on short Latin-script outputs).

## Experiments (Core vs Archived)

Core experiments in `experiments/` and exposed via `run.py`:
- Exp1 `exp1_feature_discovery.py`
- Exp3 `exp3_hindi_urdu_fixed.py`
- Exp4 `exp4_spillover.py` (includes EN→DE control)
- Exp5 `exp5_hierarchical.py`
- Exp6 `exp6_script_semantics_controls.py`
- Exp8 `exp8_scaling_9b_low_resource.py` (sanity check)
- Exp9 `exp9_layer_sweep_steering.py`
- Exp10 `exp10_attribution_occlusion.py`
- Exp11 `exp11_judge_calibration.py`
- Exp12 `exp12_qa_degradation.py`
- Exp13 `exp13_script_semantic_ablation.py`
- Exp14 `exp14_language_agnostic_space.py`

Archived (sanity/exploratory) lives under `archive/experiments_sanity/`.

## Dataset notes (rigor)

- FLORES: `load_flores()` tries `facebook/flores` and falls back to `openlanguagedata/flores_plus`.
  - In `load_research_data()`, FLORES `dev` is used as train-like data and `devtest` as held-out validation.
- MLQA: if `facebook/mlqa` fails (Hub scripts disabled), we fall back to the parquet mirror `AkshitaS/facebook_mlqa_plus`.
- Indic QA: attempts `ai4bharat/IndicQA`; if scripts are disabled, falls back to streaming `ai4bharat/Indic-Rag-Suite` per-language.
