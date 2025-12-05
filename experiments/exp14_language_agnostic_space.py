"""Experiment 14: Layer-wise Cross-Lingual Alignment (Language-Agnostic Space)

Goal
----
Empirically test whether Gemma has a more language-agnostic representation
region for English + Indic languages by measuring cross-lingual alignment
of sentence embeddings across layers.

Design
------
We use FLORES-200 parallel sentences for:
    en (English), hi (Hindi), ur (Urdu), bn (Bengali),
    ta (Tamil), te (Telugu)

For each layer ℓ in TARGET_LAYERS:
    1. For each language L and each sentence index i up to N:
         - Compute hidden states h_ℓ(L, i, t) for all tokens t.
         - Compute a sentence embedding via mean pooling:
               e_ℓ(L, i) = mean_t h_ℓ(L, i, t)
    2. For each language pair (L1, L2):
         - Compute cosine similarity cos(e_ℓ(L1, i), e_ℓ(L2, i)) for all i
         - Average over i to obtain layer-wise alignment score.

If a "language-agnostic" region exists, we expect cross-lingual cosine
similarities (e.g., EN-HI, EN-UR, EN-BN, EN-TA, EN-TE) to be lower in
early layers, peak in some mid layers, and then drop again toward late
layers where generation is more language-specific.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List
import json

import torch
import numpy as np
from tqdm import tqdm

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY
from data import load_flores
from model import GemmaWithSAE


LANGS = ["en", "hi", "ur", "bn", "ta", "te"]


def compute_sentence_embeddings(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    max_sentences: int,
) -> Dict[str, torch.Tensor]:
    """Compute mean-pooled sentence embeddings for each language at a layer.

    Returns:
        Dict lang -> tensor of shape (N, d_model) with N <= max_sentences
        aligned across languages (same indices = same FLORES sentence).
    """
    # Determine N: min length across languages so sentences align
    lengths = {lang: len(texts) for lang, texts in texts_by_lang.items()}
    n = min(min(lengths.values()), max_sentences)
    if n == 0:
        raise ValueError("No sentences available for some languages.")

    embeddings: Dict[str, List[torch.Tensor]] = {lang: [] for lang in LANGS}

    for i in tqdm(range(n), desc=f"Layer {layer} embeddings", leave=False):
        for lang in LANGS:
            text = texts_by_lang[lang][i]
            h = model.get_hidden_states(text, layer)  # (seq_len, d_model)
            sent_emb = h.mean(dim=0)  # (d_model,)
            embeddings[lang].append(sent_emb.detach().cpu())

    # Stack into (N, d_model)
    emb_tensors = {
        lang: torch.stack(vecs, dim=0) for lang, vecs in embeddings.items()
    }
    return emb_tensors


def compute_pairwise_cosine(
    emb_tensors: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute average pairwise cosine similarities between languages.

    Returns:
        Dict with keys "L1-L2" and float average cosine over sentences.
    """
    langs = sorted(emb_tensors.keys())
    # Normalize embeddings
    normed = {
        lang: torch.nn.functional.normalize(t, dim=1) for lang, t in emb_tensors.items()
    }
    pairwise: Dict[str, float] = {}
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if j <= i:
                continue
            e1 = normed[l1]
            e2 = normed[l2]
            # cos per sentence i, average
            cos_vals = (e1 * e2).sum(dim=1).numpy()
            pairwise[f"{l1}-{l2}"] = float(np.mean(cos_vals))
    return pairwise


def main():
    print("=" * 60)
    print("EXPERIMENT 14: Layer-wise Cross-Lingual Alignment")
    print("=" * 60)
    print("Testing for a more language-agnostic region across EN+Indic.\n")

    # Load model
    model = GemmaWithSAE()
    model.load_model()

    # Load FLORES data for selected languages
    print("Loading FLORES-200 parallel data...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    texts_by_lang = {lang: flores.get(lang, []) for lang in LANGS}

    # Basic sanity check
    for lang in LANGS:
        n = len(texts_by_lang[lang])
        print(f"  {lang}: {n} sentences")
        if n == 0:
            print("ERROR: Missing FLORES data for language:", lang)
            return

    max_sentences = min(500, N_SAMPLES_DISCOVERY)

    results = {
        "languages": LANGS,
        "max_sentences": max_sentences,
        "layers": {},
    }

    for layer in TARGET_LAYERS:
        print(f"\n--- Layer {layer} ---")
        emb_tensors = compute_sentence_embeddings(
            model, texts_by_lang, layer, max_sentences=max_sentences
        )
        pairwise = compute_pairwise_cosine(emb_tensors)
        results["layers"][str(layer)] = {
            "pairwise_cosine": pairwise,
        }
        print("  Example EN-HI cosine:", pairwise.get("en-hi", None))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp14_language_agnostic_space.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()

