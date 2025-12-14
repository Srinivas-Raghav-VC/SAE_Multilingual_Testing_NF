"""Shared steering utilities.

These helpers implement the core SAE / dense steering vector construction and
feature-selection logic used across experiments. They were originally defined
inside Exp2, but are experiment-agnostic and needed by multiple core runs.

Keeping them here avoids treating Exp2 as a required "sanity experiment" while
preserving a single source of truth for steering mechanics.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from tqdm import tqdm


def get_activation_diff_features(
    model,
    texts_src: List[str],
    texts_tgt: List[str],
    layer: int,
    top_k: int,
    sample_size: int = 200,
) -> List[int]:
    """Select SAE features by activation difference between target and source.

    Positive differences correspond to features more active for the target
    language.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not texts_src or not texts_tgt:
        raise ValueError(
            f"Need non-empty source/target texts for activation-diff features "
            f"(src={len(texts_src)}, tgt={len(texts_tgt)})."
        )
    model.load_sae(layer)

    src_acts = []
    for text in tqdm(texts_src[:sample_size], desc="SRC activations", leave=False):
        acts = model.get_sae_activations(text, layer)
        src_acts.append(acts.mean(dim=0).detach())
    if not src_acts:
        raise ValueError("No source activations computed (empty after slicing).")
    src_mean = torch.stack(src_acts).mean(dim=0)
    del src_acts

    tgt_acts = []
    for text in tqdm(texts_tgt[:sample_size], desc="TGT activations", leave=False):
        acts = model.get_sae_activations(text, layer)
        tgt_acts.append(acts.mean(dim=0).detach())
    if not tgt_acts:
        raise ValueError("No target activations computed (empty after slicing).")
    tgt_mean = torch.stack(tgt_acts).mean(dim=0)
    del tgt_acts

    diff = tgt_mean - src_mean
    _, top_ids = diff.topk(top_k)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return top_ids.tolist()


def get_monolinguality_features(
    model,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    target_lang: str,
    top_k: int,
    sample_size: int = 200,
) -> List[int]:
    """Select SAE features by monolinguality score for a target language."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if target_lang not in texts_by_lang:
        raise ValueError(f"target_lang '{target_lang}' not present in texts_by_lang.")
    from experiments.exp1_feature_discovery import (
        compute_activation_rates,
        compute_monolinguality,
    )

    rates_by_lang: Dict[str, torch.Tensor] = {}
    for lang, texts in texts_by_lang.items():
        if not texts:
            continue
        rates_by_lang[lang] = compute_activation_rates(model, texts[:sample_size], layer)
    if target_lang not in rates_by_lang:
        raise ValueError(
            f"No texts available for target_lang '{target_lang}' after filtering empties."
        )

    mono = compute_monolinguality(rates_by_lang, target_lang)
    _, top_ids = mono.topk(top_k)
    return top_ids.tolist()


def get_random_features(sae, top_k: int, seed: int = 42) -> List[int]:
    """Select random SAE features (negative control) without touching global RNG."""
    gen = torch.Generator(device=sae.W_dec.device if sae is not None else None)
    gen.manual_seed(seed)
    n_features = sae.cfg.d_sae
    return torch.randperm(n_features, generator=gen)[:top_k].tolist()


def construct_sae_steering_vector(model, layer: int, feature_ids: List[int]) -> torch.Tensor:
    """Build a hidden-space steering vector from SAE decoder columns."""
    if not feature_ids:
        raise ValueError("feature_ids is empty; cannot construct SAE steering vector.")
    sae = model.load_sae(layer)
    directions = sae.W_dec[feature_ids, :]  # (k, d_model)
    vector = directions.mean(dim=0)
    norm = vector.norm()
    if not torch.isfinite(norm) or norm < 1e-8:
        raise ValueError(
            f"Invalid SAE steering vector (norm={float(norm)}). "
            "This usually indicates degenerate/empty feature selection."
        )
    vector = vector / norm * (sae.cfg.d_in ** 0.5)
    return vector


def construct_dense_steering_vector(
    model,
    texts_src: List[str],
    texts_tgt: List[str],
    layer: int,
    sample_size: int = 100,
) -> torch.Tensor:
    """Build dense steering vector (mean residual activation difference)."""
    if not texts_src or not texts_tgt:
        raise ValueError(
            f"Need non-empty source/target texts for dense steering "
            f"(src={len(texts_src)}, tgt={len(texts_tgt)})."
        )
    src_hidden = []
    for text in tqdm(texts_src[:sample_size], desc="SRC hidden", leave=False):
        h = model.get_hidden_states(text, layer)
        src_hidden.append(h.mean(dim=0).detach())
    if not src_hidden:
        raise ValueError("No source hidden states computed (empty after slicing).")
    src_mean = torch.stack(src_hidden).mean(dim=0)
    del src_hidden

    tgt_hidden = []
    for text in tqdm(texts_tgt[:sample_size], desc="TGT hidden", leave=False):
        h = model.get_hidden_states(text, layer)
        tgt_hidden.append(h.mean(dim=0).detach())
    if not tgt_hidden:
        raise ValueError("No target hidden states computed (empty after slicing).")
    tgt_mean = torch.stack(tgt_hidden).mean(dim=0)
    del tgt_hidden

    vector = tgt_mean - src_mean
    norm = vector.norm()
    if not torch.isfinite(norm) or norm < 1e-8:
        raise ValueError(
            f"Invalid dense steering vector (norm={float(norm)}). "
            "Source/target representations may be identical or empty."
        )
    vector = vector / norm * (vector.shape[0] ** 0.5)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vector
