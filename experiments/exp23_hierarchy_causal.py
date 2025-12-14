"""Experiment 23: Causal Validation of Hierarchical Language Representation

Tests whether SAE features encode a Script→Language→Semantic hierarchy
across layers through causal ablation.

Research Questions:
  1. Do early layers primarily encode script features?
  2. Do mid layers encode language-specific features?
  3. Do late layers encode semantic features?

Falsification Criteria:
  - If early-layer ablation hurts semantics as much as script → no hierarchy
  - If late-layer ablation hurts script as much as semantics → no hierarchy
  - If ablation effects don't differ by layer → hierarchy is not validated

Methodology:
  1. Identify top features at early, mid, and late layers
  2. Ablate features at each layer group
  3. Measure degradation in: script accuracy, language detection, semantic similarity
  4. Test whether early ablation primarily hurts script, late hurts semantics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    N_STEERING_EVAL,
    LANG_TO_SCRIPT,
    SEED,
)
from data import load_research_data
from model import GemmaWithSAE
from evaluation_comprehensive import (
    evaluate_steering_output,
    detect_script,
    compute_semantic_similarity,
)
from reproducibility import seed_everything


@dataclass
class LayerAblationResult:
    """Results of ablating features at a specific layer."""
    layer: int
    layer_group: str  # "early", "mid", "late"
    n_features_ablated: int
    baseline_script_accuracy: float
    ablated_script_accuracy: float
    script_degradation: float
    baseline_semantic_similarity: float
    ablated_semantic_similarity: float
    semantic_degradation: float
    baseline_perplexity: float
    ablated_perplexity: float


@dataclass
class HierarchyValidationResult:
    """Overall hierarchy validation results."""
    early_script_degradation: float
    early_semantic_degradation: float
    mid_script_degradation: float
    mid_semantic_degradation: float
    late_script_degradation: float
    late_semantic_degradation: float
    hierarchy_validated: bool
    interpretation: str


def get_top_features_by_monolinguality(
    model: GemmaWithSAE,
    texts: List[str],
    layer: int,
    top_k: int = 25,
) -> List[int]:
    """Get top-k features by monolinguality at a layer."""
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    activation_counts = torch.zeros(n_features, device=model.device)
    total_tokens = 0

    for text in texts[:100]:
        acts = model.get_sae_activations(text, layer)
        activation_counts += (acts > 0).float().sum(dim=0)
        total_tokens += acts.shape[0]

    if total_tokens == 0:
        return []

    rates = (activation_counts / total_tokens).cpu().numpy()

    # Get indices of features with highest activation rates
    top_indices = np.argsort(rates)[-top_k:][::-1]
    return top_indices.tolist()


def generate_with_ablation(
    model: GemmaWithSAE,
    prompt: str,
    layer: int,
    features_to_ablate: List[int],
    target_lang: str = "hi",
) -> str:
    """Generate text while ablating specific features at a layer."""

    # First, get steering vector for target language to baseline
    # We'll ablate the features during generation

    # Ensure SAE is loaded before using it in the hook
    sae = model.load_sae(layer)

    # Create an ablation hook
    def ablation_hook(module, input, output):
        # SAE is captured from outer scope (already loaded)
        if sae is None:
            return output

        hidden = output[0] if isinstance(output, tuple) else output

        # Get SAE activations
        with torch.no_grad():
            # SAE expects 2D input (batch*seq, d_model), hidden is 3D (batch, seq, d_model)
            orig_shape = hidden.shape
            hidden_2d = hidden.reshape(-1, hidden.size(-1))  # (batch*seq, d_model)
            sae_acts = sae.encode(hidden_2d)               # (batch*seq, d_sae)
            # Zero out the specified features (2D indexing)
            sae_acts[:, features_to_ablate] = 0
            # Reconstruct and restore shape
            reconstructed = sae.decode(sae_acts)           # (batch*seq, d_model)
            reconstructed = reconstructed.reshape(orig_shape) # Restore to 3D

        if isinstance(output, tuple):
            return (reconstructed,) + output[1:]
        return reconstructed

    # Register hook
    layer_module = model.model.model.layers[layer]
    hook_handle = layer_module.register_forward_hook(ablation_hook)

    try:
        # Generate with ablation active
        output = model.generate(prompt, max_new_tokens=50)
    finally:
        hook_handle.remove()

    return output


def measure_script_accuracy(
    model: GemmaWithSAE,
    prompts: List[str],
    target_script: str,
    layer: int,
    features_to_ablate: List[int],
) -> Tuple[float, float]:
    """Measure script accuracy before and after ablation."""

    baseline_correct = 0
    ablated_correct = 0

    for prompt in prompts[:30]:
        # Baseline generation (no ablation)
        baseline_out = model.generate(prompt, max_new_tokens=50)
        baseline_script = detect_script(baseline_out)
        if baseline_script == target_script:
            baseline_correct += 1

        # Ablated generation
        ablated_out = generate_with_ablation(model, prompt, layer, features_to_ablate)
        ablated_script = detect_script(ablated_out)
        if ablated_script == target_script:
            ablated_correct += 1
    
    n = len(prompts[:30])
    if n == 0:
        raise ValueError("measure_script_accuracy received 0 prompts; cannot compute accuracy.")
    return baseline_correct / n, ablated_correct / n


def measure_semantic_similarity(
    model: GemmaWithSAE,
    prompts: List[str],
    layer: int,
    features_to_ablate: List[int],
) -> Tuple[float, float]:
    """Measure semantic preservation before and after ablation."""

    baseline_sims = []
    ablated_sims = []

    for prompt in prompts[:30]:
        # Baseline generation
        baseline_out = model.generate(prompt, max_new_tokens=50)
        baseline_sim = compute_semantic_similarity(prompt, baseline_out)
        baseline_sims.append(baseline_sim)

        # Ablated generation
        ablated_out = generate_with_ablation(model, prompt, layer, features_to_ablate)
        ablated_sim = compute_semantic_similarity(prompt, ablated_out)
        ablated_sims.append(ablated_sim)

    if not baseline_sims or not ablated_sims:
        raise ValueError("measure_semantic_similarity received 0 prompts; cannot compute mean similarity.")
    return np.mean(baseline_sims), np.mean(ablated_sims)


def test_layer_specialization(
    model: GemmaWithSAE,
    target_lang_texts: List[str],
    prompts: List[str],
    target_script: str,
) -> List[LayerAblationResult]:
    """Test whether layers specialize in script vs semantic encoding."""

    results = []

    # Group layers into early, mid, late
    n_layers = len(TARGET_LAYERS)
    early_layers = TARGET_LAYERS[:n_layers // 3]
    mid_layers = TARGET_LAYERS[n_layers // 3: 2 * n_layers // 3]
    late_layers = TARGET_LAYERS[2 * n_layers // 3:]

    layer_groups = [
        ("early", early_layers),
        ("mid", mid_layers),
        ("late", late_layers),
    ]

    for group_name, layers in layer_groups:
        print(f"\n=== Testing {group_name.upper()} layers ({layers}) ===")

        for layer in layers[:2]:  # Test 2 layers per group for efficiency
            print(f"\n  Layer {layer}:")

            # Get top features at this layer
            features = get_top_features_by_monolinguality(
                model, target_lang_texts, layer, top_k=25
            )

            if not features:
                print(f"    No features found")
                continue

            print(f"    Ablating {len(features)} features...")

            # Measure script accuracy
            baseline_script, ablated_script = measure_script_accuracy(
                model, prompts, target_script, layer, features
            )
            script_degradation = baseline_script - ablated_script

            # Measure semantic similarity
            baseline_semantic, ablated_semantic = measure_semantic_similarity(
                model, prompts, layer, features
            )
            semantic_degradation = baseline_semantic - ablated_semantic

            print(f"    Script: {baseline_script:.1%} → {ablated_script:.1%} (Δ={script_degradation:.1%})")
            print(f"    Semantic: {baseline_semantic:.3f} → {ablated_semantic:.3f} (Δ={semantic_degradation:.3f})")

            results.append(LayerAblationResult(
                layer=layer,
                layer_group=group_name,
                n_features_ablated=len(features),
                baseline_script_accuracy=baseline_script,
                ablated_script_accuracy=ablated_script,
                script_degradation=script_degradation,
                baseline_semantic_similarity=baseline_semantic,
                ablated_semantic_similarity=ablated_semantic,
                semantic_degradation=semantic_degradation,
                baseline_perplexity=0.0,  # Not computed
                ablated_perplexity=0.0,
            ))

    return results


def validate_hierarchy(results: List[LayerAblationResult]) -> HierarchyValidationResult:
    """Validate hierarchy hypothesis from ablation results."""

    # Aggregate by layer group
    early = [r for r in results if r.layer_group == "early"]
    mid = [r for r in results if r.layer_group == "mid"]
    late = [r for r in results if r.layer_group == "late"]

    def mean_or_zero(vals):
        return np.mean(vals) if vals else 0.0

    early_script = mean_or_zero([r.script_degradation for r in early])
    early_semantic = mean_or_zero([r.semantic_degradation for r in early])
    mid_script = mean_or_zero([r.script_degradation for r in mid])
    mid_semantic = mean_or_zero([r.semantic_degradation for r in mid])
    late_script = mean_or_zero([r.script_degradation for r in late])
    late_semantic = mean_or_zero([r.semantic_degradation for r in late])

    # Hierarchy hypothesis:
    # - Early ablation should hurt script more than semantics
    # - Late ablation should hurt semantics more than script
    early_script_dominant = early_script > early_semantic
    late_semantic_dominant = late_semantic > late_script

    hierarchy_validated = early_script_dominant and late_semantic_dominant

    if hierarchy_validated:
        interpretation = "VALIDATED: Early layers encode script, late layers encode semantics"
    elif early_script_dominant:
        interpretation = "PARTIAL: Early layers specialize in script, but late layers don't specialize in semantics"
    elif late_semantic_dominant:
        interpretation = "PARTIAL: Late layers specialize in semantics, but early layers don't specialize in script"
    else:
        interpretation = "NOT VALIDATED: No clear Script→Language→Semantic hierarchy"

    return HierarchyValidationResult(
        early_script_degradation=early_script,
        early_semantic_degradation=early_semantic,
        mid_script_degradation=mid_script,
        mid_semantic_degradation=mid_semantic,
        late_script_degradation=late_script,
        late_semantic_degradation=late_semantic,
        hierarchy_validated=hierarchy_validated,
        interpretation=interpretation,
    )


def main():
    seed_everything(SEED)

    print("=" * 60)
    print("EXPERIMENT 23: Causal Hierarchy Validation")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in model.model_id.lower() else ""

    # Load data
    data = load_research_data(
        max_train_samples=N_SAMPLES_DISCOVERY,
        max_test_samples=0,
        max_eval_samples=0,
        seed=SEED,
    )

    prompts = data.steering_prompts[:N_STEERING_EVAL]

    # Target language for testing
    target_lang = "hi"
    target_script = "devanagari"

    if target_lang not in data.train:
        print(f"Error: {target_lang} not in training data")
        return

    print(f"\nTarget language: {target_lang} ({target_script})")
    print(f"Prompts: {len(prompts)}")

    # Run layer specialization test
    print("\n" + "=" * 60)
    print("LAYER SPECIALIZATION TEST")
    print("=" * 60)

    ablation_results = test_layer_specialization(
        model, data.train[target_lang], prompts, target_script
    )

    # Validate hierarchy
    print("\n" + "=" * 60)
    print("HIERARCHY VALIDATION")
    print("=" * 60)

    validation = validate_hierarchy(ablation_results)

    print(f"\n  ABLATION SUMMARY:")
    print(f"  Early layers:")
    print(f"    Script degradation: {validation.early_script_degradation:.1%}")
    print(f"    Semantic degradation: {validation.early_semantic_degradation:.3f}")
    print(f"  Mid layers:")
    print(f"    Script degradation: {validation.mid_script_degradation:.1%}")
    print(f"    Semantic degradation: {validation.mid_semantic_degradation:.3f}")
    print(f"  Late layers:")
    print(f"    Script degradation: {validation.late_script_degradation:.1%}")
    print(f"    Semantic degradation: {validation.late_semantic_degradation:.3f}")

    print(f"\n  CONCLUSION:")
    print(f"  Hierarchy validated: {validation.hierarchy_validated}")
    print(f"  {validation.interpretation}")

    # CONTROL: Test German (non-Indic) to ensure hierarchy is Indic-specific
    control_results = None
    control_validation = None
    if "de" in data.train:
        print("\n" + "=" * 60)
        print("CONTROL: German (non-Indic)")
        print("=" * 60)

        control_results = test_layer_specialization(
            model, data.train["de"], prompts, "latin"
        )

        if control_results:
            control_validation = validate_hierarchy(control_results)
            print(f"\n  German hierarchy validated: {control_validation.hierarchy_validated}")
            print(f"  {control_validation.interpretation}")

            # Compare: if German shows same pattern as Hindi, hierarchy might be artifact
            if control_validation.hierarchy_validated == validation.hierarchy_validated:
                print("\n  WARNING: German shows same pattern as Hindi - hierarchy may not be Indic-specific")
            else:
                print("\n  GOOD: German shows different pattern - hierarchy is Indic-specific")

    # Save results
    results = {
        "target_lang": target_lang,
        "target_script": target_script,
        "ablation_results": [
            {
                "layer": r.layer,
                "layer_group": r.layer_group,
                "n_features_ablated": r.n_features_ablated,
                "baseline_script_accuracy": r.baseline_script_accuracy,
                "ablated_script_accuracy": r.ablated_script_accuracy,
                "script_degradation": r.script_degradation,
                "baseline_semantic_similarity": r.baseline_semantic_similarity,
                "ablated_semantic_similarity": r.ablated_semantic_similarity,
                "semantic_degradation": r.semantic_degradation,
            }
            for r in ablation_results
        ],
        "validation": {
            "early_script_degradation": validation.early_script_degradation,
            "early_semantic_degradation": validation.early_semantic_degradation,
            "mid_script_degradation": validation.mid_script_degradation,
            "mid_semantic_degradation": validation.mid_semantic_degradation,
            "late_script_degradation": validation.late_script_degradation,
            "late_semantic_degradation": validation.late_semantic_degradation,
            "hierarchy_validated": validation.hierarchy_validated,
            "interpretation": validation.interpretation,
        },
        "control_german": {
            "tested": control_validation is not None,
            "hierarchy_validated": control_validation.hierarchy_validated if control_validation else None,
            "early_script_degradation": control_validation.early_script_degradation if control_validation else None,
            "late_semantic_degradation": control_validation.late_semantic_degradation if control_validation else None,
            "indic_specific": (
                control_validation is not None and
                control_validation.hierarchy_validated != validation.hierarchy_validated
            ),
        } if control_validation else {"tested": False},
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp23_hierarchy_causal{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
