"""Data loading for multilingual SAE experiments."""

from datasets import load_dataset
from config import LANGUAGES


def load_flores(split="devtest", max_samples=None):
    """Load FLORES-200 parallel sentences."""
    data = {}
    for short_code, flores_code in LANGUAGES.items():
        ds = load_dataset("facebook/flores", flores_code, split=split, trust_remote_code=True)
        sentences = ds["sentence"]
        if max_samples:
            sentences = sentences[:max_samples]
        data[short_code] = sentences
    return data


def load_parallel_pairs(lang1, lang2, max_samples=500):
    """Load parallel sentence pairs."""
    flores = load_flores(max_samples=max_samples)
    return list(zip(flores[lang1], flores[lang2]))


if __name__ == "__main__":
    # Quick test
    flores = load_flores(max_samples=3)
    for lang, sents in flores.items():
        print(f"{lang}: {sents[0][:60]}...")
