"""Data loading for SAE Multilingual Steering Research.

Datasets:
1. AI4Bharat Samanantar - 49.6M parallel sentences for 11 Indic languages
2. FLORES-200 - 200 languages, smaller but high quality
3. MLQA - Multilingual QA for evaluation (EN, HI, DE, AR, ES, VI, ZH)
4. IndicQA - Indic-specific QA for evaluation

Train/Test Split Strategy:
- Training data (feature discovery): Samanantar or FLORES train split
- Validation data: FLORES test split or held-out
- Evaluation data: MLQA/IndicQA (completely separate!)
"""

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from datasets import load_dataset

from config import LANGUAGES, SEED, EVAL_PROMPTS
from reproducibility import seed_everything


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataSplit:
    """Container for properly split data with strict separation guarantees.

    Critical for research validity:
    - train: Used ONLY for computing steering vectors / feature statistics
    - test: Used ONLY for evaluating feature metrics (held-out validation)
    - steering_prompts: Used ONLY for steering evaluation (must not overlap with train)
    - qa_eval: Completely separate QA datasets (MLQA, IndicQA)
    """
    train: Dict[str, List[str]]           # For feature discovery / steering vector computation
    test: Dict[str, List[str]]            # For feature validation (held-out)
    steering_prompts: List[str]           # For steering evaluation
    qa_eval: Optional[Dict[str, Any]] = None  # For QA evaluation
    data_provenance: Optional[Dict[str, Any]] = None  # Track data sources

    def verify_no_leakage(self) -> Tuple[bool, Dict[str, int]]:
        """Verify train and test don't overlap.

        Returns:
            Tuple of (passed: bool, overlap_counts: Dict[str, int])
        """
        overlap_counts = {}
        all_clean = True

        for lang in self.train.keys():
            if lang not in self.test:
                continue
            train_set = set(self.train[lang])
            test_set = set(self.test[lang])
            overlap = train_set & test_set
            if overlap:
                overlap_counts[lang] = len(overlap)
                print(f"WARNING: {len(overlap)} sentences overlap in {lang}!")
                all_clean = False

        # Also check steering prompts against train
        train_all = set()
        for sents in self.train.values():
            train_all.update(sents)

        steering_overlap = set(self.steering_prompts) & train_all
        if steering_overlap:
            overlap_counts["steering_vs_train"] = len(steering_overlap)
            print(f"WARNING: {len(steering_overlap)} steering prompts overlap with training data!")
            all_clean = False

        return all_clean, overlap_counts

    def get_fingerprint(self) -> str:
        """Generate a hash fingerprint of the data split for reproducibility."""
        import hashlib

        content = []
        for lang in sorted(self.train.keys()):
            content.append(f"train_{lang}_{len(self.train[lang])}")
            if self.train[lang]:
                content.append(self.train[lang][0][:50])
        for lang in sorted(self.test.keys()):
            content.append(f"test_{lang}_{len(self.test[lang])}")
        content.append(f"steering_{len(self.steering_prompts)}")

        return hashlib.md5("".join(content).encode()).hexdigest()[:12]

    def summary(self) -> str:
        """Generate a summary string for logging."""
        lines = ["DataSplit Summary:"]
        lines.append(f"  Fingerprint: {self.get_fingerprint()}")
        lines.append(f"  Train languages: {list(self.train.keys())}")
        lines.append(f"  Test languages: {list(self.test.keys())}")
        for lang in self.train:
            lines.append(f"    {lang}: {len(self.train.get(lang, []))} train, {len(self.test.get(lang, []))} test")
        lines.append(f"  Steering prompts: {len(self.steering_prompts)}")
        if self.qa_eval:
            lines.append(f"  QA datasets: {list(self.qa_eval.keys())}")
        return "\n".join(lines)


@dataclass  
class LanguageData:
    """Data for a single language."""
    code: str
    name: str
    script: str
    family: str
    sentences: List[str] = field(default_factory=list)


# =============================================================================
# LANGUAGE METADATA
# =============================================================================

LANGUAGE_METADATA = {
    # Indic - Indo-Aryan
    "hi": {"name": "Hindi", "script": "Devanagari", "family": "Indo-Aryan"},
    "ur": {"name": "Urdu", "script": "Arabic", "family": "Indo-Aryan"},
    "bn": {"name": "Bengali", "script": "Bengali", "family": "Indo-Aryan"},
    "mr": {"name": "Marathi", "script": "Devanagari", "family": "Indo-Aryan"},
    "gu": {"name": "Gujarati", "script": "Gujarati", "family": "Indo-Aryan"},
    "pa": {"name": "Punjabi", "script": "Gurmukhi", "family": "Indo-Aryan"},
    "or": {"name": "Odia", "script": "Odia", "family": "Indo-Aryan"},
    "as": {"name": "Assamese", "script": "Assamese", "family": "Indo-Aryan"},
    
    # Indic - Dravidian
    "ta": {"name": "Tamil", "script": "Tamil", "family": "Dravidian"},
    "te": {"name": "Telugu", "script": "Telugu", "family": "Dravidian"},
    "kn": {"name": "Kannada", "script": "Kannada", "family": "Dravidian"},
    "ml": {"name": "Malayalam", "script": "Malayalam", "family": "Dravidian"},
    
    # European
    "en": {"name": "English", "script": "Latin", "family": "Germanic"},
    "de": {"name": "German", "script": "Latin", "family": "Germanic"},
    "fr": {"name": "French", "script": "Latin", "family": "Romance"},
    "es": {"name": "Spanish", "script": "Latin", "family": "Romance"},
    
    # Other
    "ar": {"name": "Arabic", "script": "Arabic", "family": "Semitic"},
    "zh": {"name": "Chinese", "script": "Han", "family": "Sino-Tibetan"},
    "ja": {"name": "Japanese", "script": "Mixed", "family": "Japonic"},
    "vi": {"name": "Vietnamese", "script": "Latin", "family": "Austroasiatic"},
}


# =============================================================================
# FLORES-200 LOADING
# =============================================================================

def load_flores(
    max_samples: Optional[int] = None,
    languages: Optional[Dict[str, str]] = None,
    split: str = "devtest"
) -> Dict[str, List[str]]:
    """Load FLORES-200 parallel sentences.
    
    Args:
        max_samples: Maximum samples per language
        languages: Dict mapping short code to FLORES code (default: from config)
        split: Dataset split ("dev" / "devtest" are ~1k sentences; exact count
            depends on dataset variant and language)
        
    Returns:
        Dict mapping language code to list of sentences
    """
    if languages is None:
        languages = LANGUAGES

    # Treat `max_samples=0` as "load nothing" (useful for quick validation runs),
    # and avoid accidentally interpreting it as "no limit" via Python truthiness.
    if max_samples is not None and max_samples <= 0:
        return {short_code: [] for short_code in languages.keys()}
    
    data = {}
    
    for short_code, flores_code in languages.items():
        try:
            # Newer versions of `datasets` deprecate `trust_remote_code` for
            # Hub-hosted datasets. We rely on the locally cached FLORES-200
            # data; if it is unavailable, this will raise and we fall back to
            # an empty list for that language.
            ds = None
            last_err = None
            for ds_name in ("facebook/flores", "openlanguagedata/flores_plus"):
                try:
                    ds = load_dataset(
                        ds_name,
                        flores_code,
                        split=split,
                    )
                    break
                except Exception as e:
                    last_err = e
                    ds = None

            if ds is None:
                raise last_err or RuntimeError("Unknown error loading FLORES.")
            sentences = list(ds["sentence"])
            
            if max_samples is not None and len(sentences) > max_samples:
                sentences = sentences[:max_samples]
            
            data[short_code] = sentences
            print(f"  Loaded {short_code}: {len(sentences)} sentences")
            
        except Exception as e:
            print(f"  Warning: Could not load {short_code} ({flores_code}): {e}")
            # Return empty list, don't crash
            data[short_code] = []
    
    # Publication-grade strictness: enforce minimum samples when STRICT_DATA=1
    if str(os.environ.get("STRICT_DATA", "0")).lower() in ("1", "true", "yes"):
        try:
            from config import MIN_LANG_SAMPLES
            for lang, sentences in data.items():
                if len(sentences) < MIN_LANG_SAMPLES:
                    raise ValueError(
                        f"STRICT_DATA=1: FLORES language '{lang}' has only {len(sentences)} "
                        f"samples (< MIN_LANG_SAMPLES={MIN_LANG_SAMPLES})."
                    )
        except ImportError:
            pass

    return data


# =============================================================================
# STEERING PROMPTS (SHARED HELPERS)
# =============================================================================

def load_steering_prompts(
    min_prompts: Optional[int] = None,
    seed: int = SEED,
) -> List[str]:
    """Build a steering/evaluation prompt list with enough diversity.

    Many experiments need more prompts than the small handcrafted EVAL_PROMPTS
    list provides. We centralize the augmentation logic here so that every
    generation-based experiment uses the same prompt pool.

    Strategy:
      1) Start from EVAL_PROMPTS (handwritten, diverse tasks).
      2) If fewer than `min_prompts`, augment with held-out FLORES English
         sentences (devtest split), deduped.

    Args:
        min_prompts: Minimum prompt count to return. Defaults to N_STEERING_EVAL.
        seed: RNG seed for reproducible prompt ordering when sampling.

    Returns:
        List of prompts (length >= min_prompts when FLORES is available).
    """
    from config import N_STEERING_EVAL

    if min_prompts is None:
        min_prompts = N_STEERING_EVAL

    random.seed(seed)
    prompts = EVAL_PROMPTS.copy()

    if len(prompts) >= min_prompts:
        return prompts

    needed = min_prompts - len(prompts)
    try:
        flores_en = load_flores(
            max_samples=needed + 50,
            languages={"en": "eng_Latn"},
            split="devtest",
        )
        extra = flores_en.get("en", [])
    except Exception as e:
        print(f"[data] Warning: could not augment prompts from FLORES: {e}")
        extra = []

    if extra:
        existing = set(prompts)
        # Deterministic order for reproducibility.
        for p in extra:
            if p in existing:
                continue
            prompts.append(p)
            existing.add(p)
            if len(prompts) >= min_prompts:
                break

    if len(prompts) < min_prompts:
        print(
            f"[data] Warning: only {len(prompts)} steering prompts available "
            f"(requested {min_prompts}). Experiments may be underpowered."
        )

    return prompts


# =============================================================================
# AI4BHARAT SAMANANTAR LOADING
# =============================================================================

def load_samanantar(
    lang: str,
    max_samples: Optional[int] = None,
    split: str = "train"
) -> Tuple[List[str], List[str]]:
    """Load AI4Bharat Samanantar parallel corpus.
    
    Samanantar has 49.6M sentence pairs between English and 11 Indic languages:
    as, bn, gu, hi, kn, ml, mr, or, pa, ta, te
    
    Args:
        lang: Target language code (e.g., "hi" for Hindi)
        max_samples: Maximum samples to load
        split: Dataset split (default "train")
        
    Returns:
        Tuple of (english_sentences, target_sentences)
    """
    if max_samples is not None and max_samples <= 0:
        return [], []

    try:
        # Prefer streaming to avoid materializing tens of millions of pairs in RAM.
        try:
            ds_stream = load_dataset("ai4bharat/samanantar", lang, split=split, streaming=True)
            en_sentences: List[str] = []
            tgt_sentences: List[str] = []
            for i, row in enumerate(ds_stream):
                if max_samples is not None and i >= max_samples:
                    break
                en_sentences.append(row.get("src", ""))
                tgt_sentences.append(row.get("tgt", ""))

            print(f"  Loaded Samanantar {lang} (streaming): {len(en_sentences)} pairs")
            return en_sentences, tgt_sentences
        except Exception:
            # Fall back to non-streaming (cached Arrow). Still avoid converting
            # the full split to a Python list before truncation.
            ds = load_dataset("ai4bharat/samanantar", lang, split=split)
            if max_samples is not None:
                ds = ds.select(range(min(int(max_samples), len(ds))))

            en_sentences = list(ds["src"])
            tgt_sentences = list(ds["tgt"])

            print(f"  Loaded Samanantar {lang}: {len(en_sentences)} pairs")
            return en_sentences, tgt_sentences
        
    except Exception as e:
        print(f"  Warning: Could not load Samanantar {lang}: {e}")
        return [], []


def load_samanantar_multilingual(
    languages: List[str],
    max_samples_per_lang: int = 5000
) -> Dict[str, List[str]]:
    """Load Samanantar for multiple languages.
    
    Args:
        languages: List of language codes (e.g., ["hi", "bn", "ta"])
        max_samples_per_lang: Max samples per language
        
    Returns:
        Dict mapping language code to sentences
    """
    data = {"en": []}  # Will collect English from all pairs
    
    for lang in languages:
        if lang == "en":
            continue
            
        en_sents, tgt_sents = load_samanantar(lang, max_samples_per_lang)
        
        if tgt_sents:
            data[lang] = tgt_sents
            # Add English sentences (may have duplicates, that's OK)
            data["en"].extend(en_sents)
    
    # Deduplicate and limit English deterministically for reproducibility.
    unique_en = sorted(set(data["en"]))
    data["en"] = unique_en[:max_samples_per_lang]
    print(f"  English (combined): {len(data['en'])} unique sentences")
    
    return data


# =============================================================================
# MLQA LOADING (FOR EVALUATION)
# =============================================================================

def load_mlqa(
    languages: Optional[List[str]] = None,
    max_samples: int = 500
) -> Dict[str, List[Dict]]:
    """Load MLQA question-answering dataset.
    
    MLQA has QA pairs in: en, ar, de, es, hi, vi, zh
    
    Args:
        languages: Languages to load (default: all available)
        max_samples: Max samples per language
        
    Returns:
        Dict mapping language to list of QA examples
    """
    available = ["en", "ar", "de", "es", "hi", "vi", "zh"]
    
    if languages is None:
        languages = available
    else:
        languages = [l for l in languages if l in available]
    
    data = {}
    
    for lang in languages:
        # Prefer scriptless/parquet mirrors when possible. Newer versions of
        # `datasets` may refuse to execute Hub dataset scripts.
        #
        # AkshitaS/facebook_mlqa_plus is a parquet mirror of facebook/mlqa.
        dataset_candidates = [
            "AkshitaS/facebook_mlqa_plus",
            "facebook/mlqa",
        ]

        # Common MLQA config names:
        # - "mlqa.en.en" style configs for monolingual QA
        # - "mlqa-translate-test.<lang>" for translated test sets
        configs = []
        if lang == "en":
            configs = ["mlqa.en.en"]
        else:
            configs = [f"mlqa-translate-test.{lang}", f"mlqa.{lang}.{lang}"]

        splits = ["test", "validation"]

        examples: List[Dict] = []
        loaded = False
        last_err: Optional[Exception] = None

        for ds_name in dataset_candidates:
            for cfg in configs:
                for split in splits:
                    try:
                        ds = load_dataset(ds_name, cfg, split=split)
                        for i, item in enumerate(ds):
                            if i >= max_samples:
                                break
                            answers = item.get("answers") or {}
                            answer_texts = answers.get("text") or []
                            answer = answer_texts[0] if answer_texts else ""
                            examples.append(
                                {
                                    "id": item.get("id", ""),
                                    "context": item.get("context", ""),
                                    "question": item.get("question", ""),
                                    "answer": answer,
                                }
                            )
                        loaded = True
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if loaded:
                    break
            if loaded:
                break

        if loaded:
            data[lang] = examples
            print(f"  Loaded MLQA {lang}: {len(examples)} QA pairs")
        else:
            print(f"  Warning: Could not load MLQA {lang}: {last_err}")
            data[lang] = []
    
    return data


# =============================================================================
# INDICQA LOADING (FOR EVALUATION)
# =============================================================================

def load_indicqa(
    languages: Optional[List[str]] = None,
    max_samples: int = 500
) -> Dict[str, List[Dict]]:
    """Load AI4Bharat IndicQA dataset.
    
    IndicQA has QA pairs in 11 Indic languages.
    
    Args:
        languages: Languages to load
        max_samples: Max samples per language
        
    Returns:
        Dict mapping language to list of QA examples
    """
    available = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
    
    if languages is None:
        languages = available
    else:
        languages = [l for l in languages if l in available]
    
    data = {}
    
    for lang in languages:
        # Primary: AI4Bharat IndicQA (script-based on the Hub; may fail on
        # `datasets` versions that disallow dataset scripts).
        try:
            ds = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}", split="test")
            examples: List[Dict] = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                answers = item.get("answers") or {}
                answer_texts = answers.get("text") or [""]
                examples.append(
                    {
                        "context": item.get("context", ""),
                        "question": item.get("question", ""),
                        "answer": answer_texts[0] if answer_texts else "",
                    }
                )
            data[lang] = examples
            print(f"  Loaded IndicQA {lang}: {len(examples)} QA pairs")
            continue
        except Exception as e:
            print(f"  Warning: Could not load IndicQA {lang}: {e}")

        # Fallback: Indic-Rag-Suite (parquet, very large). We stream and
        # take the first `max_samples` examples for a lightweight, reliable
        # QA-style evaluation set.
        #
        # This keeps the evaluation in the Indic domain (context+question+answer)
        # without requiring Hub dataset scripts.
        try:
            ds = load_dataset(
                "ai4bharat/Indic-Rag-Suite",
                lang,
                split="train",
                streaming=True,
            )
            examples = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                ctx = item.get("paragraph") or item.get("context") or ""
                q = item.get("question") or ""
                a = item.get("answer") or ""
                examples.append({"context": ctx, "question": q, "answer": a})
            data[lang] = examples
            print(
                f"  Loaded Indic-Rag-Suite fallback {lang}: {len(examples)} QA pairs"
            )
        except Exception as e2:
            print(f"  Warning: Could not load Indic-Rag-Suite {lang}: {e2}")
            data[lang] = []
    
    return data


# =============================================================================
# COMBINED DATA LOADING WITH TRAIN/TEST SPLIT
# =============================================================================

def load_research_data(
    max_train_samples: int = 5000,
    max_test_samples: int = 1000,
    max_eval_samples: int = 500,
    use_samanantar: bool = True,
    seed: int = SEED
) -> DataSplit:
    """Load complete research dataset with proper splits.
    
    Strategy:
    - Training: Samanantar (large) or FLORES (train portion)
    - Validation: FLORES (test portion) 
    - Evaluation: MLQA + steering prompts (completely separate!)
    
    Args:
        max_train_samples: Max samples for training per language
        max_test_samples: Max samples for testing per language
        max_eval_samples: Max samples for evaluation per language
        use_samanantar: Use Samanantar for training (recommended)
        seed: Random seed
        
    Returns:
        DataSplit with train, test, steering_prompts, qa_eval
    """
    # Seed once here so data splits and prompt augmentation are reproducible
    # even when this helper is called outside run.py / experiment mains.
    seed_everything(seed)
    
    print("\n" + "=" * 60)
    print("LOADING RESEARCH DATA")
    print("=" * 60)
    
    # Languages for different purposes
    indic_langs = ["hi", "ur", "bn", "ta", "te"]  # Core Indic
    control_langs = ["en", "de", "ar"]  # Non-Indic controls
    all_langs = indic_langs + control_langs
    
    # ----- TRAINING DATA -----
    print("\n1. Loading TRAINING data...")
    
    if use_samanantar:
        # Use Samanantar for Indic languages (much larger!)
        # Includes all 4 Dravidian: ta, te, kn, ml for complete coverage
        samanantar_langs = ["hi", "bn", "ta", "te", "kn", "ml"]  # Samanantar languages
        train_data = load_samanantar_multilingual(samanantar_langs, max_train_samples)
        
        # Add control languages from FLORES
        # Use FLORES dev split for "train-like" data, and devtest for held-out
        # validation. This avoids the degenerate case where train/test both
        # come from devtest and then are fully deduped away.
        flores_train = load_flores(
            max_train_samples,
            {
                "de": "deu_Latn",
                "ar": "arb_Arab",
            },
            split="dev",
        )
        train_data.update(flores_train)
        
        # Add Urdu from FLORES (not in Samanantar)
        flores_ur = load_flores(
            max_train_samples,
            {"ur": "urd_Arab"},
            split="dev",
        )
        train_data.update(flores_ur)
        if str(os.environ.get("STRICT_DATA", "0")).lower() in ("1", "true", "yes"):
            from config import MIN_LANG_SAMPLES
            for lang, samples in train_data.items():
                if len(samples) < MIN_LANG_SAMPLES:
                    raise ValueError(
                        f"STRICT_DATA=1: Training data for '{lang}' has only {len(samples)} samples "
                        f"(< MIN_LANG_SAMPLES={MIN_LANG_SAMPLES})."
                    )
    else:
        # Use FLORES for everything (smaller but simpler)
        flores_codes = {
            "en": "eng_Latn",
            "hi": "hin_Deva",
            "ur": "urd_Arab",
            "bn": "ben_Beng",
            "ta": "tam_Taml",
            "te": "tel_Telu",
            "kn": "kan_Knda",  # Kannada (Dravidian)
            "ml": "mal_Mlym",  # Malayalam (Dravidian)
            "de": "deu_Latn",
            "ar": "arb_Arab",
        }
        train_data = load_flores(max_train_samples, flores_codes, split="dev")

    # ----- TEST DATA -----
    print("\n2. Loading TEST data (validation)...")

    # Always use FLORES for test (held-out from training if using FLORES)
    # Includes all Dravidian languages for complete cluster analysis
    flores_codes = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "ur": "urd_Arab",
        "bn": "ben_Beng",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "kn": "kan_Knda",  # Kannada (Dravidian)
        "ml": "mal_Mlym",  # Malayalam (Dravidian)
        "de": "deu_Latn",
        "ar": "arb_Arab",
    }
    test_data = load_flores(max_test_samples, flores_codes, split="devtest")
    
    # Ensure train / test are disjoint wherever possible to avoid leakage.
    # We do this regardless of whether Samanantar or FLORES is used for
    # training: any sentence that appears in the test split is removed from
    # the corresponding training split.
    for lang in list(train_data.keys()):
        if lang in test_data:
            test_set = set(test_data[lang])
            if not test_set:
                continue
            orig_len = len(train_data[lang])
            train_data[lang] = [s for s in train_data[lang] if s not in test_set]
            if len(train_data[lang]) < orig_len:
                removed = orig_len - len(train_data[lang])
                print(f"  [dedupe] Removed {removed} overlapping {lang} sentences from train.")
    
    # ----- EVALUATION DATA -----
    print("\n3. Loading EVALUATION data (QA)...")
    
    qa_eval = {}
    
    # MLQA for Hindi, German, Arabic (cross-lingual comparison)
    mlqa_data = load_mlqa(["en", "hi", "de", "ar"], max_eval_samples)
    qa_eval["mlqa"] = mlqa_data
    
    # IndicQA for more Indic languages (all Dravidian + key Indo-Aryan)
    indicqa_data = load_indicqa(["hi", "bn", "ta", "te", "kn", "ml"], max_eval_samples)
    qa_eval["indicqa"] = indicqa_data
    
    # ----- STEERING PROMPTS -----
    print("\n4. Loading steering prompts...")
    steering_prompts = EVAL_PROMPTS.copy()
    
    # Augment with FLORES English sentences to reach N_STEERING_EVAL
    # We use FLORES 'devtest' which is our standard validation/test set.
    # Using it for steering evaluation is safe (held-out from training).
    from config import N_STEERING_EVAL
    
    if len(steering_prompts) < N_STEERING_EVAL:
        needed = N_STEERING_EVAL - len(steering_prompts)
        print(f"  Augmenting with {needed} FLORES English sentences...")
        
        # Load enough FLORES English sentences
        # We assume 'eng_Latn' is available via load_flores
        flores_en = load_flores(
            max_samples=needed + 50, 
            languages={"en": "eng_Latn"}
        )
        
        if flores_en and "en" in flores_en:
            extra_prompts = flores_en["en"]
            # Filter out any that might duplicate EVAL_PROMPTS (unlikely but good practice)
            existing_set = set(steering_prompts)
            added_count = 0
            for p in extra_prompts:
                if p not in existing_set:
                    steering_prompts.append(p)
                    added_count += 1
                if added_count >= needed:
                    break
            print(f"  Added {added_count} prompts from FLORES")

    print(f"  {len(steering_prompts)} steering prompts total")

    if str(os.environ.get("STRICT_DATA", "0")).lower() in ("1", "true", "yes"):
        try:
            from config import MIN_STEERING_PROMPTS
            if len(steering_prompts) < MIN_STEERING_PROMPTS:
                raise ValueError(
                    f"STRICT_DATA=1: Only {len(steering_prompts)} steering prompts "
                    f"(< MIN_STEERING_PROMPTS={MIN_STEERING_PROMPTS})."
                )
        except ImportError:
            pass
    
    # ----- CREATE DATASPLIT -----
    # Publication-grade provenance: record what data sources/splits were used.
    provenance: Dict[str, Any] = {
        "train": {},
        "test": {},
        "steering_prompts": {
            "n_total": len(steering_prompts),
            "n_handcrafted": len(EVAL_PROMPTS),
            "n_augmented_flores_en": max(0, len(steering_prompts) - len(EVAL_PROMPTS)),
            "source": "EVAL_PROMPTS + FLORES(en, devtest) augmentation",
        },
        "qa_eval": {
            "mlqa": {
                "langs": {k: len(v) for k, v in (mlqa_data or {}).items()},
                "source": "facebook/mlqa or parquet mirror (see load_mlqa)",
                "splits": ["test", "validation"],
            },
            "indicqa": {
                "langs": {k: len(v) for k, v in (indicqa_data or {}).items()},
                "source": "ai4bharat/IndicQA (test) with fallback ai4bharat/Indic-Rag-Suite (train, streaming)",
            },
        },
    }

    if use_samanantar:
        for lang in train_data:
            if lang in {"en", "hi", "bn", "ta", "te", "kn", "ml"}:
                provenance["train"][lang] = "ai4bharat/samanantar (train)"
            elif lang in {"ur", "de", "ar"}:
                provenance["train"][lang] = "FLORES (dev)"
            else:
                provenance["train"][lang] = "unknown"
    else:
        for lang in train_data:
            provenance["train"][lang] = "FLORES (dev)"

    for lang in test_data:
        provenance["test"][lang] = "FLORES (devtest)"

    data_split = DataSplit(
        train=train_data,
        test=test_data,
        steering_prompts=steering_prompts,
        qa_eval=qa_eval,
        data_provenance=provenance,
    )
    
    # ----- VERIFY -----
    print("\n5. Verifying data integrity...")
    no_leakage, overlap_counts = data_split.verify_no_leakage()
    if no_leakage:
        print("  ✓ No data leakage detected")
    else:
        print("  ✗ WARNING: Potential data leakage!")
        for key, count in overlap_counts.items():
            print(f"     - {key}: {count} overlapping items")
        if str(os.environ.get("STRICT_DATA", "0")).lower() in ("1", "true", "yes"):
            raise ValueError("STRICT_DATA=1: data leakage detected; refusing to continue.")

    print(f"\n  Data fingerprint: {data_split.get_fingerprint()}")
    
    # ----- SUMMARY -----
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Training languages: {list(train_data.keys())}")
    print(f"Test languages: {list(test_data.keys())}")
    for lang in train_data:
        train_n = len(train_data.get(lang, []))
        test_n = len(test_data.get(lang, []))
        print(f"  {lang}: {train_n} train, {test_n} test")
    print(f"MLQA evaluation: {list(qa_eval.get('mlqa', {}).keys())}")
    print(f"IndicQA evaluation: {list(qa_eval.get('indicqa', {}).keys())}")
    print("=" * 60)
    
    return data_split


# =============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# =============================================================================

def load_parallel_pairs(
    lang1: str,
    lang2: str,
    max_samples: int = 500
) -> List[Tuple[str, str]]:
    """Load parallel sentence pairs (legacy function)."""
    flores = load_flores(max_samples)
    
    if lang1 not in flores or lang2 not in flores:
        return []
    
    return list(zip(flores[lang1], flores[lang2]))


# =============================================================================
# MAIN (TESTING)
# =============================================================================

if __name__ == "__main__":
    print("Testing data loading...")
    
    # Test basic FLORES loading
    print("\n--- Testing FLORES ---")
    flores = load_flores(max_samples=10)
    for lang, sents in flores.items():
        if sents:
            print(f"{lang}: {sents[0][:50]}...")
    
    # Test research data loading
    print("\n--- Testing Full Research Data ---")
    # Use smaller samples for testing
    data = load_research_data(
        max_train_samples=100,
        max_test_samples=20,
        max_eval_samples=10,
        use_samanantar=False  # Use FLORES for quick test
    )
    
    print("\nTest passed!")
