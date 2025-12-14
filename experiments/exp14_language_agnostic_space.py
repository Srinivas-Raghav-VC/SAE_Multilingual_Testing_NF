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
import os

import torch
import numpy as np
from tqdm import tqdm

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY, SEED
from data import load_flores
from model import GemmaWithSAE
from reproducibility import seed_everything


LANGS = ["en", "hi", "ur", "bn", "ta", "te"]


def compute_sentence_embeddings_all_layers(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layers: List[int],
    max_sentences: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Compute mean-pooled sentence embeddings for each language at multiple layers.

    This is significantly cheaper than recomputing embeddings separately for
    each layer: for each FLORES sentence we do one forward pass and collect the
    mean-pooled hidden states for all requested layers.

    Returns:
        Dict layer -> (Dict lang -> tensor of shape (N, d_model))
        aligned across languages (same indices = same FLORES sentence).
    """
    lengths = {lang: len(texts) for lang, texts in texts_by_lang.items()}
    n = min(min(lengths.values()), max_sentences)
    if n == 0:
        raise ValueError("No sentences available for some languages.")

    emb_lists: Dict[int, Dict[str, List[torch.Tensor]]] = {
        layer: {lang: [] for lang in LANGS} for layer in layers
    }

    # Determine hidden_states indexing convention once.
    try:
        n_blocks = len(model.model.model.layers)  # type: ignore[attr-defined]
    except Exception:
        n_blocks = None

    for i in tqdm(range(n), desc="FLORES sentence embeddings", leave=False):
        for lang in LANGS:
            text = texts_by_lang[lang][i]
            inputs = model.tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            if hs is None:
                raise RuntimeError("Model did not return hidden_states.")

            # HF convention: hidden_states[0] is embedding output and
            # hidden_states[layer+1] corresponds to transformer block `layer`.
            if n_blocks is not None and len(hs) == n_blocks + 1:
                base = 1
            else:
                base = 0

            for layer in layers:
                idx = layer + base
                if not (0 <= idx < len(hs)):
                    raise ValueError(
                        f"Requested layer {layer} (mapped to hidden_states[{idx}]) "
                        f"out of range for hidden_states length {len(hs)}."
                    )
                h = hs[idx].squeeze(0)  # (seq, d_model)
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                emb_lists[layer][lang].append(h.mean(dim=0).detach().cpu())

    emb_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        emb_tensors[layer] = {lang: torch.stack(v, dim=0) for lang, v in emb_lists[layer].items()}
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
    seed_everything(SEED)
    print("=" * 60)
    print("EXPERIMENT 14: Layer-wise Cross-Lingual Alignment")
    print("=" * 60)
    print("Testing for a more language-agnostic region across EN+Indic.\n")

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in str(getattr(model, "model_id", "")).lower() else ""

    # Load FLORES data for selected languages
    print("Loading FLORES-200 parallel data...")
    # Use FLORES dev for representation analysis to avoid overlapping with
    # devtest prompts used elsewhere for evaluation augmentation.
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY, split="dev")
    texts_by_lang = {lang: flores.get(lang, []) for lang in LANGS}

    # Basic sanity check
    lengths = {}
    for lang in LANGS:
        n = len(texts_by_lang[lang])
        lengths[lang] = n
        print(f"  {lang}: {n} sentences")
        if n == 0:
            print("ERROR: Missing FLORES data for language:", lang)
            return
    # For FLORES dev/test splits the sentences are parallel by index; we
    # assert this here so that cross-lingual alignment is meaningful. If
    # this assertion fails, it likely indicates a data loading issue.
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise RuntimeError(
            f"[exp14] Expected parallel FLORES sets with equal length, "
            f"but got lengths={lengths}"
        )

    # Default to a moderate N to keep this experiment feasible on shared GPUs.
    # Set N_ALIGNMENT_SENTENCES to increase/decrease.
    max_sentences = int(os.environ.get("N_ALIGNMENT_SENTENCES", "200"))
    max_sentences = max(10, min(max_sentences, 500, N_SAMPLES_DISCOVERY))

    results = {
        "languages": LANGS,
        "max_sentences": max_sentences,
        "layers": {},
    }

    layers = list(TARGET_LAYERS)
    print(f"Computing embeddings for layers={layers} ...")
    emb_by_layer = compute_sentence_embeddings_all_layers(
        model,
        texts_by_lang,
        layers=layers,
        max_sentences=max_sentences,
    )

    for layer in layers:
        pairwise = compute_pairwise_cosine(emb_by_layer[layer])
        results["layers"][str(layer)] = {"pairwise_cosine": pairwise}
        print(f"  Layer {layer}: Example EN-HI cosine:", pairwise.get("en-hi", None))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp14_language_agnostic_space{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
