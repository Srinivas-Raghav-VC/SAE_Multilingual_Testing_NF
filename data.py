"""Data loading for multilingual SAE experiments.

UPDATED: Uses openlanguagedata/flores_plus (FLORES+ v4.2) instead of 
deprecated facebook/flores. The new dataset is gated and requires HF login.

Changes from facebook/flores:
- Dataset ID: openlanguagedata/flores_plus
- Column name: "text" instead of "sentence"
- Uses ISO 639-3 + script codes (e.g., "hin_Deva", "eng_Latn")
- Requires Hugging Face authentication (gated dataset)
"""

import os
import sys
from datasets import load_dataset
from config import LANGUAGES


def check_hf_login():
    """Check if HF_TOKEN is set or user is logged in."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("✓ HF_TOKEN found in environment")
        return True
    
    # Check for cached login
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ HF login found (cached token)")
            return True
    except Exception:
        pass
    
    print("⚠ WARNING: No HF authentication found!")
    print("  FLORES+ is a gated dataset requiring login.")
    print("  Options:")
    print("    1. export HF_TOKEN=your_token_here")
    print("    2. huggingface-cli login")
    print("    3. Accept terms at: https://huggingface.co/datasets/openlanguagedata/flores_plus")
    return False


def load_flores(split="devtest", max_samples=None):
    """Load FLORES+ parallel sentences.
    
    FLORES+ v4.2 (openlanguagedata/flores_plus) replaces deprecated facebook/flores.
    This is a gated dataset - requires HF authentication.
    
    Args:
        split: "dev" or "devtest" (FLORES+ splits)
        max_samples: Max sentences per language
        
    Returns:
        Dict mapping short lang code -> list of sentences
    """
    check_hf_login()
    
    data = {}
    for short_code, flores_code in LANGUAGES.items():
        try:
            # FLORES+ uses language subsets (e.g., "hin_Deva")
            ds = load_dataset(
                "openlanguagedata/flores_plus",
                flores_code,
                split=split,
                trust_remote_code=True
            )
            # FLORES+ uses "text" column instead of "sentence"
            sentences = ds["text"]
            if max_samples:
                sentences = sentences[:max_samples]
            data[short_code] = sentences
            print(f"  Loaded {len(sentences)} {short_code} sentences")
        except Exception as e:
            print(f"  ERROR loading {short_code} ({flores_code}): {e}")
            raise
    
    return data


def load_parallel_pairs(lang1, lang2, max_samples=500):
    """Load parallel sentence pairs for two languages.
    
    FLORES+ sentences are aligned by 'id' field across languages.
    """
    flores = load_flores(max_samples=max_samples)
    return list(zip(flores[lang1], flores[lang2]))


if __name__ == "__main__":
    print("Testing FLORES+ data loading...")
    print("=" * 50)
    
    if not check_hf_login():
        print("\nPlease authenticate before running experiments.")
        sys.exit(1)
    
    print("\nLoading sample data...")
    flores = load_flores(max_samples=3)
    for lang, sents in flores.items():
        print(f"\n{lang}: {sents[0][:60]}...")
