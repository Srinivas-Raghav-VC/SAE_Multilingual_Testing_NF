"""Experiment 13: Group Ablation for Script-Sensitive vs Script-Invariant Features

Goal
-----
Give causal backing to the operational split between:
  - script-sensitive features (e.g., HI script-only),
  - script-invariant / semantic features (HI semantic),
by ablating *groups* of features and measuring how script ratios,
semantic similarity, and degradation change.

Design
------
We reuse the Exp6 setup to define, at a chosen layer:
  - hi_script_only: active on Hindi Devanagari and noise, but not on
    transliterated Hindi or Urdu.
  - hi_semantic: active on both Hindi Devanagari and transliterated Hindi.

Then we:
  1) Generate baseline outputs on EVAL_PROMPTS.
  2) Ablate all hi_script_only features and re-run generation.
  3) Ablate all hi_semantic features and re-run generation.

We compare:
  - Δ script ratio (target Devanagari),
  - Δ semantic similarity (LaBSE, using prompt as reference),
  - Δ degradation rate (repetition).

If script-only group ablation mainly changes script ratio with small
semantic impact, while semantic group ablation hurts semantic similarity
more, this supports the "script-sensitive vs script-invariant" labels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import List, Set, Dict
import os
import random

import torch
from tqdm import tqdm

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY, EVAL_PROMPTS, SEED
from data import load_flores, load_steering_prompts
from model import GemmaWithSAE
from evaluation_comprehensive import evaluate_steering_output, aggregate_results
from experiments.exp10_attribution_occlusion import ablate_feature_in_hidden
from reproducibility import seed_everything


def simple_transliterate_hi_to_latin(text: str) -> str:
    """Rough transliteration of Hindi to Latin (copied from Exp6)."""
    mapping = {
        "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
        "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
        "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
        "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
        "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
        "य": "y", "र": "r", "ल": "l", "व": "v",
        "श": "sh", "ष": "sh", "स": "s", "ह": "h",
        "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
        "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
        "ं": "n", "ँ": "n", "ः": "h",
    }
    out = []
    for ch in text:
        out.append(mapping.get(ch, ch))
    return "".join(out)


def generate_devanagari_noise(n: int, length: int = 20) -> List[str]:
    """Generate random Devanagari character noise strings (copied from Exp6)."""
    chars = [chr(c) for c in range(0x0900, 0x097F)]
    rng = random.Random(int(os.environ.get("GROUP_ABLATION_SEED", "42")))
    samples = []
    for _ in range(n):
        s = "".join(rng.choice(chars) for _ in range(length))
        samples.append(s)
    return samples


def compute_activation_mask(
    model: GemmaWithSAE,
    texts: List[str],
    layer: int,
    threshold: float = 0.01,
) -> torch.Tensor:
    """Compute boolean mask for features active above threshold (copied from Exp6)."""
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae
    counts = torch.zeros(n_features, device=model.device)
    total_tokens = 0

    for text in texts:
        acts = model.get_sae_activations(text, layer)
        counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]

    rates = counts / max(total_tokens, 1)
    return rates > threshold


def get_hi_script_and_semantic_sets(
    model: GemmaWithSAE,
    hi_texts: List[str],
    ur_texts: List[str],
    bn_texts: List[str],
    layer: int,
    rate_threshold: float = 0.01,
) -> Dict[str, Set[int]]:
    """Reproduce Exp6's hi_script_only and hi_semantic sets for a layer."""
    # Base Hindi in Devanagari
    hi_mask = compute_activation_mask(model, hi_texts, layer, rate_threshold)

    # Transliterated Hindi (Latin script, similar semantics)
    hi_trans = [simple_transliterate_hi_to_latin(t) for t in hi_texts[:200]]
    hi_trans_mask = compute_activation_mask(model, hi_trans, layer, rate_threshold)

    # Devanagari noise
    noise_texts = generate_devanagari_noise(min(len(hi_texts), 500))
    noise_mask = compute_activation_mask(model, noise_texts, layer, rate_threshold)

    # Urdu and Bengali (not used directly in group definitions here,
    # but can be used for future cross-language checks)
    _ = compute_activation_mask(model, ur_texts, layer, rate_threshold)
    _ = compute_activation_mask(model, bn_texts, layer, rate_threshold)

    hi_feats = set(int(i) for i in hi_mask.nonzero(as_tuple=False).flatten().tolist())
    hi_trans_feats = set(int(i) for i in hi_trans_mask.nonzero(as_tuple=False).flatten().tolist())
    noise_feats = set(int(i) for i in noise_mask.nonzero(as_tuple=False).flatten().tolist())

    # Script-only: active on Devanagari (HI or noise) but not on transliteration
    script_candidates = hi_feats | noise_feats
    script_excluded = hi_trans_feats
    hi_script_only = script_candidates - script_excluded

    # Semantic within Hindi: active on both HI_Deva and HI_Latin
    hi_semantic = hi_feats & hi_trans_feats

    return {
        "hi_script_only": hi_script_only,
        "hi_semantic": hi_semantic,
    }


def ablate_group_in_hidden(
    sae,
    hidden: torch.Tensor,
    feature_indices: List[int],
) -> torch.Tensor:
    """Ablate a group of SAE features via encode->zero->decode."""
    if not feature_indices:
        return hidden
    b, s, d = hidden.shape
    hs = hidden.reshape(-1, d)
    with torch.no_grad():
        z = sae.encode(hs)  # (b*s, d_sae)
        z[:, feature_indices] = 0.0
        h_abl = sae.decode(z)  # (b*s, d_model)
    # Align dtype/device to avoid silent bf16/fp32 mismatches downstream.
    h_abl = h_abl.to(hidden.dtype).to(hidden.device)
    return h_abl.reshape(b, s, d)


def run_group_ablation(
    model: GemmaWithSAE,
    layer: int,
    group_indices: List[int],
    prompts: List[str],
    group_name: str,
) -> Dict[str, float]:
    """Compare baseline vs group-ablated metrics for a set of prompts."""
    if not group_indices:
        raise ValueError(f"{group_name}: feature group is empty; cannot run ablation.")
    sae = model.load_sae(layer)

    # Baseline
    base_results = []
    for p in prompts:
        out = model.generate(p, max_new_tokens=64)
        base_results.append(
            evaluate_steering_output(
                p,
                out,
                method="baseline",
                strength=0.0,
                layer=layer,
                compute_semantics=True,
                use_llm_judge=False,
            )
        )
    base_agg = aggregate_results(base_results)

    # Group ablation via forward hook
    abl_results = []

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_abl = ablate_group_in_hidden(sae, h, group_indices)
        return (h_abl,) + output[1:] if isinstance(output, tuple) else h_abl

    handle = model.model.model.layers[layer].register_forward_hook(hook)
    try:
        for p in prompts:
            out = model.generate(p, max_new_tokens=64)
            abl_results.append(
                evaluate_steering_output(
                    p,
                    out,
                    method=f"{group_name}_ablated",
                    strength=0.0,
                    layer=layer,
                    compute_semantics=True,
                    use_llm_judge=False,
                )
            )
    finally:
        handle.remove()

    abl_agg = aggregate_results(abl_results)

    return {
        "base_script_ratio": base_agg.avg_target_script_ratio,
        "base_semantic_similarity": base_agg.avg_semantic_similarity or 0.0,
        "base_degradation_rate": base_agg.degradation_rate,
        "base_sensitivity": base_agg.sensitivity,
        "abl_script_ratio": abl_agg.avg_target_script_ratio,
        "abl_semantic_similarity": abl_agg.avg_semantic_similarity or 0.0,
        "abl_degradation_rate": abl_agg.degradation_rate,
        "abl_sensitivity": abl_agg.sensitivity,
        "delta_script_ratio": (abl_agg.avg_target_script_ratio - base_agg.avg_target_script_ratio),
        "delta_semantic_similarity": (abl_agg.avg_semantic_similarity or 0.0)
        - (base_agg.avg_semantic_similarity or 0.0),
        "delta_degradation_rate": abl_agg.degradation_rate - base_agg.degradation_rate,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 13: Group Ablation for Script vs Semantic Features")
    print("=" * 60)

    seed_everything(SEED)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in str(getattr(model, "model_id", "")).lower() else ""

    # Load FLORES data for HI/UR/BN
    print("\nLoading FLORES data...")
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY)
    hi_texts = flores.get("hi", [])
    ur_texts = flores.get("ur", [])
    bn_texts = flores.get("bn", [])

    if not hi_texts or not ur_texts or not bn_texts:
        print("ERROR: Need hi, ur, bn in FLORES data.")
        return

    hi_texts = hi_texts[:1000]
    ur_texts = ur_texts[:1000]
    bn_texts = bn_texts[:1000]

    # Choose a representative layer (late-ish, where steering is strong)
    layer_candidates = [l for l in TARGET_LAYERS if 10 <= l <= 24]
    layer = layer_candidates[-1] if layer_candidates else TARGET_LAYERS[-1]
    print(f"\nUsing layer {layer} for group ablation analysis.")

    # Define groups
    groups = get_hi_script_and_semantic_sets(model, hi_texts, ur_texts, bn_texts, layer)
    hi_script_only = sorted(groups["hi_script_only"])
    hi_semantic = sorted(groups["hi_semantic"])

    print(f"  #HI script-only features: {len(hi_script_only)}")
    print(f"  #HI semantic (script-robust) features: {len(hi_semantic)}")

    # ------------------------------------------------------------------
    # Causal negative controls: random feature groups (size-matched)
    # ------------------------------------------------------------------
    # Reviewer-proofing: group ablations can look impressive even when "any big
    # group of features" causes disruption. We therefore ablate random feature
    # groups of the same size as each group, and report their deltas alongside
    # the targeted groups.
    seed = int(os.environ.get("GROUP_ABLATION_SEED", "42"))
    max_features = int(os.environ.get("GROUP_ABLATION_MAX_FEATURES", "0"))
    rng = random.Random(seed)

    def _maybe_cap(group: List[int]) -> List[int]:
        if max_features and len(group) > max_features:
            capped = sorted(rng.sample(group, k=max_features))
            print(
                f"[exp13] Note: capping group size from {len(group)} -> {len(capped)} "
                f"(GROUP_ABLATION_MAX_FEATURES={max_features})."
            )
            return capped
        return group

    hi_script_only = _maybe_cap(hi_script_only)
    hi_semantic = _maybe_cap(hi_semantic)

    sae = model.load_sae(layer)
    n_features = int(sae.cfg.d_sae)
    excluded = set(hi_script_only) | set(hi_semantic)
    candidates = [i for i in range(n_features) if i not in excluded]
    rng.shuffle(candidates)
    need_script = len(hi_script_only)
    need_sem = len(hi_semantic)
    if len(candidates) < (need_script + need_sem):
        print(
            "[exp13] Warning: not enough non-group features to form size-matched "
            "random baselines without overlap; sampling from all features."
        )
        candidates = list(range(n_features))
        rng.shuffle(candidates)

    rand_script_only = sorted(candidates[:need_script]) if need_script else []
    rand_semantic = (
        sorted(candidates[need_script : need_script + need_sem]) if need_sem else []
    )

    # Use a larger prompt pool for stable group-ablation effects.
    all_prompts = load_steering_prompts(min_prompts=120)
    n_prompts = int(os.environ.get("N_GROUP_ABLATION_PROMPTS", "100"))
    prompts = all_prompts[:n_prompts]

    print("\nRunning group ablation for HI script-only features...")
    script_res = run_group_ablation(
        model,
        layer=layer,
        group_indices=hi_script_only,
        prompts=prompts,
        group_name="hi_script_only",
    )

    print("\nRunning group ablation for HI semantic features...")
    semantic_res = run_group_ablation(
        model,
        layer=layer,
        group_indices=hi_semantic,
        prompts=prompts,
        group_name="hi_semantic",
    )

    random_script_res = None
    if rand_script_only:
        print("\nRunning random (size-matched) ablation for HI script-only group...")
        random_script_res = run_group_ablation(
            model,
            layer=layer,
            group_indices=rand_script_only,
            prompts=prompts,
            group_name="random_script_only",
        )

    random_sem_res = None
    if rand_semantic:
        print("\nRunning random (size-matched) ablation for HI semantic group...")
        random_sem_res = run_group_ablation(
            model,
            layer=layer,
            group_indices=rand_semantic,
            prompts=prompts,
            group_name="random_semantic",
        )

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    import json

    out_path = out_dir / f"exp13_script_semantic_ablation{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "layer": layer,
                "n_hi_script_only": len(hi_script_only),
                "n_hi_semantic": len(hi_semantic),
                "group_ablation_seed": seed,
                "group_ablation_max_features": max_features,
                "n_random_script_only": len(rand_script_only),
                "n_random_semantic": len(rand_semantic),
                "hi_script_only_ablation": script_res,
                "hi_semantic_ablation": semantic_res,
                "random_script_only_ablation": random_script_res,
                "random_semantic_ablation": random_sem_res,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
