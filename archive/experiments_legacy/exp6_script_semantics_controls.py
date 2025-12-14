"""Experiment 6: Script vs Semantics Controls for Indic Languages

Goal:
    More cleanly separate script-only features from semantic/generative
    features for Indic languages using controlled conditions:
      - Hindi in Devanagari vs transliterated Hindi (Latin script)
      - Script-only noise in Devanagari
      - Cross-language same-script vs same-semantics comparisons

This extends the Hindi–Urdu analysis by:
    1. Adding within-language transliteration controls
    2. Identifying features that:
        - React to script only
        - React to semantics irrespective of script
        - Act as language-family or Hindi-specific semantics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Dict, List, Set
import random

import torch

from config import TARGET_LAYERS, N_SAMPLES_DISCOVERY
from data import load_flores
from evaluation_comprehensive import jaccard_overlap
from model import GemmaWithSAE


@dataclass
class ScriptSemanticsResult:
    layer: int
    hi_script_only: Set[int]
    hi_semantic: Set[int]
    family_semantic: Set[int]
    noise_script_only: Set[int]
    jaccard_hi_deva_hi_latin: float
    jaccard_hi_deva_noise: float
    jaccard_hi_deva_ur: float
    jaccard_hi_deva_bn: float


def simple_transliterate_hi_to_latin(text: str) -> str:
    """Very rough transliteration of Hindi to Latin.

    This is intentionally simple and lossy; we only need a control
    where semantics stay roughly the same but the script changes.
    """
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
    """Generate random Devanagari character noise strings."""
    chars = [chr(c) for c in range(0x0900, 0x097F)]
    samples = []
    for _ in range(n):
        s = "".join(random.choice(chars) for _ in range(length))
        samples.append(s)
    return samples


def compute_activation_mask(model: GemmaWithSAE, texts: List[str], layer: int, threshold: float = 0.01) -> torch.Tensor:
    """Compute boolean mask for features active above threshold."""
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


def analyze_layer(
    model: GemmaWithSAE,
    hi_texts: List[str],
    ur_texts: List[str],
    bn_texts: List[str],
    layer: int,
    rate_threshold: float = 0.01,
) -> ScriptSemanticsResult:
    """Analyze one layer for script vs semantic features."""
    print(f"\n=== Layer {layer} ===")

    # Base Hindi in Devanagari
    hi_mask = compute_activation_mask(model, hi_texts, layer, rate_threshold)

    # Transliterated Hindi (Latin script, similar semantics)
    hi_trans = [simple_transliterate_hi_to_latin(t) for t in hi_texts[:200]]
    hi_trans_mask = compute_activation_mask(model, hi_trans, layer, rate_threshold)

    # Devanagari noise
    noise_texts = generate_devanagari_noise(min(len(hi_texts), 500))
    noise_mask = compute_activation_mask(model, noise_texts, layer, rate_threshold)

    # Urdu (same spoken language, Arabic script)
    ur_mask = compute_activation_mask(model, ur_texts, layer, rate_threshold)

    # Bengali (same family, different script)
    bn_mask = compute_activation_mask(model, bn_texts, layer, rate_threshold)

    # Convert to sets of feature indices
    hi_feats = set(int(i) for i in hi_mask.nonzero(as_tuple=False).flatten().tolist())
    hi_trans_feats = set(int(i) for i in hi_trans_mask.nonzero(as_tuple=False).flatten().tolist())
    noise_feats = set(int(i) for i in noise_mask.nonzero(as_tuple=False).flatten().tolist())
    ur_feats = set(int(i) for i in ur_mask.nonzero(as_tuple=False).flatten().tolist())
    bn_feats = set(int(i) for i in bn_mask.nonzero(as_tuple=False).flatten().tolist())

    print(f"  Hindi (Deva): {len(hi_feats)} active")
    print(f"  Hindi (Latin): {len(hi_trans_feats)} active")
    print(f"  Devanagari noise: {len(noise_feats)} active")
    print(f"  Urdu: {len(ur_feats)} active")
    print(f"  Bengali: {len(bn_feats)} active")

    # Script-only: active on Devanagari (HI or noise) but not on transliteration or Urdu
    script_candidates = hi_feats | noise_feats
    script_excluded = hi_trans_feats | ur_feats
    hi_script_only = script_candidates - script_excluded

    # Semantic within Hindi: active on both HI_Deva and HI_Latin
    hi_semantic = hi_feats & hi_trans_feats

    # Family semantics: active on HI + UR + BN (Indo-Aryan family)
    family_semantic = hi_feats & ur_feats & bn_feats

    # Noise-specific script: active on noise but not on real Hindi
    noise_script_only = noise_feats - hi_feats

    # Sanity: Jaccard scores
    j_hi_latin = float(jaccard_overlap(hi_feats, hi_trans_feats))
    j_hi_noise = float(jaccard_overlap(hi_feats, noise_feats))
    j_hi_ur = float(jaccard_overlap(hi_feats, ur_feats))
    j_hi_bn = float(jaccard_overlap(hi_feats, bn_feats))
    print("  Jaccard(HI_Deva, HI_Latin): {:.1%}".format(j_hi_latin))
    print("  Jaccard(HI_Deva, Noise): {:.1%}".format(j_hi_noise))
    print("  Jaccard(HI_Deva, UR): {:.1%}".format(j_hi_ur))
    print("  Jaccard(HI_Deva, BN): {:.1%}".format(j_hi_bn))

    print(f"  HI script-only features: {len(hi_script_only)}")
    print(f"  HI semantic (script-robust) features: {len(hi_semantic)}")
    print(f"  Family semantic features (HI+UR+BN): {len(family_semantic)}")
    print(f"  Noise-only Devanagari features: {len(noise_script_only)}")

    return ScriptSemanticsResult(
        layer=layer,
        hi_script_only=hi_script_only,
        hi_semantic=hi_semantic,
        family_semantic=family_semantic,
        noise_script_only=noise_script_only,
        jaccard_hi_deva_hi_latin=j_hi_latin,
        jaccard_hi_deva_noise=j_hi_noise,
        jaccard_hi_deva_ur=j_hi_ur,
        jaccard_hi_deva_bn=j_hi_bn,
    )


def main():
    print("=" * 60)
    print("EXPERIMENT 6: Script vs Semantics Controls")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in str(getattr(model, "model_id", "")).lower() else ""

    # Load FLORES data for key languages
    print("\nLoading FLORES data...")
    # Use FLORES dev for feature estimation; devtest is reserved for evaluation
    # prompts in steering experiments.
    flores = load_flores(max_samples=N_SAMPLES_DISCOVERY, split="dev")
    hi_texts = flores.get("hi", [])
    ur_texts = flores.get("ur", [])
    bn_texts = flores.get("bn", [])

    if not hi_texts or not ur_texts or not bn_texts:
        print("ERROR: Need hi, ur, bn in FLORES data.")
        return

    # Use a subset for speed
    hi_texts = hi_texts[:1000]
    ur_texts = ur_texts[:1000]
    bn_texts = bn_texts[:1000]

    results = {}
    for layer in TARGET_LAYERS:
        res = analyze_layer(model, hi_texts, ur_texts, bn_texts, layer)
        results[layer] = {
            "hi_script_only": len(res.hi_script_only),
            "hi_semantic": len(res.hi_semantic),
            "family_semantic": len(res.family_semantic),
            "noise_script_only": len(res.noise_script_only),
            "jaccard_hi_deva_hi_latin": res.jaccard_hi_deva_hi_latin,
            "jaccard_hi_deva_noise": res.jaccard_hi_deva_noise,
            "jaccard_hi_deva_ur": res.jaccard_hi_deva_ur,
            "jaccard_hi_deva_bn": res.jaccard_hi_deva_bn,
        }

    # Save summary
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    import json

    out_path = out_dir / f"exp6_script_semantics_controls{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
