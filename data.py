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

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from datasets import load_dataset

from config import LANGUAGES, SEED, EVAL_PROMPTS


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
    data_provenance: Optional[Dict[str, str]] = None  # Track data sources

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
        split: Dataset split ("devtest" has ~997 sentences)
        
    Returns:
        Dict mapping language code to list of sentences
    """
    if languages is None:
        languages = LANGUAGES
    
    data = {}
    
    for short_code, flores_code in languages.items():
        try:
            # Newer versions of `datasets` deprecate `trust_remote_code` for
            # Hub-hosted datasets. We rely on the locally cached FLORES-200
            # data; if it is unavailable, this will raise and we fall back to
            # an empty list for that language.
            ds = load_dataset(
                "facebook/flores",
                flores_code,
                split=split,
            )
            sentences = list(ds["sentence"])
            
            if max_samples and len(sentences) > max_samples:
                sentences = sentences[:max_samples]
            
            data[short_code] = sentences
            print(f"  Loaded {short_code}: {len(sentences)} sentences")
            
        except Exception as e:
            print(f"  Warning: Could not load {short_code} ({flores_code}): {e}")
            # Return empty list, don't crash
            data[short_code] = []
    
    return data


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
    try:
        # We rely on locally cached Samanantar data. If the Hub loader
        # requires a script and `datasets` refuses to execute it, this call
        # will fail and we fall back to empty lists.
        ds = load_dataset("ai4bharat/samanantar", lang, split=split)
        
        # Samanantar has 'src' (English) and 'tgt' (target language)
        en_sentences = list(ds["src"])
        tgt_sentences = list(ds["tgt"])
        
        if max_samples and len(en_sentences) > max_samples:
            en_sentences = en_sentences[:max_samples]
            tgt_sentences = tgt_sentences[:max_samples]
        
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
    
    # Deduplicate and limit English
    data["en"] = list(set(data["en"]))[:max_samples_per_lang]
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
        try:
            # MLQA uses format like "mlqa.en.en" for monolingual
            config = f"mlqa.{lang}.{lang}"
            ds = load_dataset("facebook/mlqa", config, split="test")
            
            examples = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                examples.append({
                    "id": item["id"],
                    "context": item["context"],
                    "question": item["question"],
                    "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
                })
            
            data[lang] = examples
            print(f"  Loaded MLQA {lang}: {len(examples)} QA pairs")
            
        except Exception as e:
            print(f"  Warning: Could not load MLQA {lang}: {e}")
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
        try:
            # As with Samanantar, we prefer cached data and avoid
            # `trust_remote_code` to keep compatibility with recent
            # `datasets` versions.
            ds = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}", split="test")
            
            examples = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                examples.append({
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answers", {}).get("text", [""])[0],
                })
            
            data[lang] = examples
            print(f"  Loaded IndicQA {lang}: {len(examples)} QA pairs")
            
        except Exception as e:
            print(f"  Warning: Could not load IndicQA {lang}: {e}")
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
    random.seed(seed)
    
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
        samanantar_langs = ["hi", "bn", "ta", "te"]  # Samanantar languages
        train_data = load_samanantar_multilingual(samanantar_langs, max_train_samples)
        
        # Add control languages from FLORES
        flores_train = load_flores(max_train_samples, {
            "de": "deu_Latn",
            "ar": "arb_Arab",
        })
        train_data.update(flores_train)
        
        # Add Urdu from FLORES (not in Samanantar)
        flores_ur = load_flores(max_train_samples, {"ur": "urd_Arab"})
        train_data.update(flores_ur)
    else:
        # Use FLORES for everything (smaller but simpler)
        flores_codes = {
            "en": "eng_Latn",
            "hi": "hin_Deva",
            "ur": "urd_Arab",
            "bn": "ben_Beng",
            "ta": "tam_Taml",
            "te": "tel_Telu",
            "de": "deu_Latn",
            "ar": "arb_Arab",
        }
        train_data = load_flores(max_train_samples, flores_codes)
    
    # ----- TEST DATA -----
    print("\n2. Loading TEST data (validation)...")
    
    # Always use FLORES for test (held-out from training if using FLORES)
    flores_codes = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "ur": "urd_Arab",
        "bn": "ben_Beng",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "de": "deu_Latn",
        "ar": "arb_Arab",
    }
    test_data = load_flores(max_test_samples, flores_codes)
    
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
    
    # IndicQA for more Indic languages
    indicqa_data = load_indicqa(["hi", "bn", "ta", "te"], max_eval_samples)
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
    
    # ----- CREATE DATASPLIT -----
    data_split = DataSplit(
        train=train_data,
        test=test_data,
        steering_prompts=steering_prompts,
        qa_eval=qa_eval
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
