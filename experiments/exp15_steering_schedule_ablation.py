"""Experiment 15: Steering Schedule Ablation

Compares different steering schedules to optimize the trade-off between
script adherence (dominance) and semantic faithfulness (quality).

Schedules tested:
1. No steering (true baseline)
2. Constant steering (standard approach)
3. Prompt-only (steer only during prompt processing)
4. Generation-only (steer only during token generation)
5. Exponential decay (decay steering strength over time)

Hypothesis:
- Generation-only improves semantic coherence/faithfulness over constant.
- Decay helps maintain script while reducing degradation.
- All steered conditions should outperform baseline on script dominance.

Statistical Analysis:
- Wilcoxon signed-rank tests between all pairs of schedules
- Holm-Bonferroni correction for multiple comparisons
- Effect sizes (rank-biserial correlation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from tqdm import tqdm
import numpy as np

from config import (
    TARGET_LAYERS,
    N_SAMPLES_EVAL,
    TARGET_LANGUAGE,
    LANG_TO_SCRIPT,
    SEED
)
from model import GemmaWithSAE
from data import load_research_data
from steering_utils import (
    get_activation_diff_features,
    construct_sae_steering_vector
)
from evaluation_comprehensive import (
    evaluate_steering_output,
    aggregate_results,
    LID_CONFIDENCE_THRESHOLD
)
from stats import wilcoxon_test, holm_bonferroni_correction
from evaluation_comprehensive import estimate_power_paired
from reproducibility import seed_everything

def run_statistical_comparisons(per_prompt_data: dict) -> dict:
    """Run pairwise Wilcoxon tests between all schedule conditions.

    Args:
        per_prompt_data: Dict[schedule_name] -> Dict[metric_name] -> List[float]

    Returns:
        Dict with pairwise comparison results and Holm-corrected p-values
    """
    schedules = list(per_prompt_data.keys())
    comparisons = []
    p_values = []

    # Compare each steered schedule against baseline
    metrics_to_compare = ["script_ratio", "semantic_sim"]

    for metric in metrics_to_compare:
        for i, sched1 in enumerate(schedules):
            for sched2 in schedules[i + 1:]:
                vals1 = per_prompt_data[sched1].get(metric, [])
                vals2 = per_prompt_data[sched2].get(metric, [])

                if len(vals1) != len(vals2) or len(vals1) < 5:
                    continue

                try:
                    result = wilcoxon_test(vals1, vals2)
                    comparisons.append({
                        "metric": metric,
                        "schedule_1": sched1,
                        "schedule_2": sched2,
                        "p_value": result.p_value,
                        "effect_size": result.effect_size,
                        "median_diff": np.median(vals1) - np.median(vals2),
                    })
                    p_values.append(result.p_value)
                except Exception as e:
                    print(f"  Warning: Wilcoxon test failed for {sched1} vs {sched2} ({metric}): {e}")

    # Apply Holm-Bonferroni correction
    if p_values:
        correction = holm_bonferroni_correction(p_values)
        for i, comp in enumerate(comparisons):
            comp["adjusted_p"] = correction["adjusted_p_values"][i]
            comp["significant"] = correction["significant"][i]
            # Approximate power for script ratio difference
            try:
                comp["power"] = estimate_power_paired(
                    per_prompt_data[comp["schedule_1"]]["script_ratio"],
                    per_prompt_data[comp["schedule_2"]]["script_ratio"],
                )
            except Exception:
                comp["power"] = None

    return {
        "comparisons": comparisons,
        "n_comparisons": len(comparisons),
        "alpha": 0.05,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 15: Steering Schedule Ablation")
    print("=" * 60)

    seed_everything(SEED)

    # 1. Setup
    model = GemmaWithSAE()
    model.load_model()

    use_9b = "9b" in model.model_id.lower()
    suffix = "_9b" if use_9b else ""

    # Pick a strong middle layer for steering
    layer = TARGET_LAYERS[len(TARGET_LAYERS) // 2]
    print(f"Targeting Layer {layer} for schedule ablation.")

    # 2. Data
    data = load_research_data(
        max_train_samples=500,
        max_test_samples=0,
        max_eval_samples=0,
        seed=SEED
    )

    prompts = data.steering_prompts[:N_SAMPLES_EVAL]
    print(f"Evaluating on {len(prompts)} prompts.")

    # 3. Construct Steering Vector (EN -> HI)
    print("\nConstructing EN -> HI steering vector...")
    tgt_lang = "hi"
    src_lang = "en"

    if tgt_lang not in data.train or src_lang not in data.train:
        print("Error: Missing training data for vector construction.")
        return

    features = get_activation_diff_features(
        model,
        texts_src=data.train[src_lang],
        texts_tgt=data.train[tgt_lang],
        layer=layer,
        top_k=25
    )

    steering_vector = construct_sae_steering_vector(model, layer, features)
    steering_strength = 2.0

    # 4. Define Conditions (including TRUE baseline with no steering)
    schedules = [
        {"name": "no_steering", "schedule": None, "decay": 1.0},  # TRUE BASELINE
        {"name": "constant", "schedule": "constant", "decay": 1.0},
        {"name": "prompt_only", "schedule": "prompt_only", "decay": 1.0},
        {"name": "generation_only", "schedule": "generation_only", "decay": 1.0},
        {"name": "decay_0.95", "schedule": "exp_decay", "decay": 0.95},
        {"name": "decay_0.90", "schedule": "exp_decay", "decay": 0.90},
        {"name": "decay_0.80", "schedule": "exp_decay", "decay": 0.80},
        {"name": "decay_0.70", "schedule": "exp_decay", "decay": 0.70},
    ]

    results = {
        "layer": layer,
        "strength": steering_strength,
        "features": features,
        "n_prompts": len(prompts),
        "schedules": {},
    }

    target_script = LANG_TO_SCRIPT[tgt_lang]

    # Store per-prompt data for statistical tests
    per_prompt_data = {}

    # 5. Run Ablation
    for cond in schedules:
        name = cond["name"]
        sched = cond["schedule"]
        decay = cond["decay"]

        print(f"\nRunning Condition: {name}")

        eval_results = []
        script_ratios = []
        semantic_sims = []

        for prompt in tqdm(prompts, desc=name):
            if sched is None:
                # TRUE BASELINE: no steering at all
                output = model.generate(prompt, max_new_tokens=100)
            else:
                output = model.generate_with_steering(
                    prompt=prompt,
                    layer=layer,
                    steering_vector=steering_vector,
                    strength=steering_strength,
                    schedule=sched,
                    decay=decay,
                    max_new_tokens=100
                )

            res = evaluate_steering_output(
                prompt=prompt,
                output=output,
                method=name,
                strength=steering_strength if sched else 0.0,
                layer=layer,
                target_script=target_script,
                semantic_reference=prompt,
                compute_semantics=True
            )
            eval_results.append(res)

            # Store per-prompt values for stats
            script_ratios.append(res.script_ratios.get(target_script, 0.0))
            semantic_sims.append(res.semantic_similarity if res.semantic_similarity >= 0 else 0.0)

        # Aggregate
        agg = aggregate_results(eval_results, target_script=target_script)

        # Store per-prompt data
        per_prompt_data[name] = {
            "script_ratio": script_ratios,
            "semantic_sim": semantic_sims,
        }

        # Store aggregate metrics
        results["schedules"][name] = {
            "schedule_type": sched,
            "decay": decay,
            "script_dominance": agg.avg_target_script_ratio,
            "success_rate": agg.success_rate,
            "semantic_similarity": agg.avg_semantic_similarity,
            "degradation_rate": agg.degradation_rate,
            "avg_repetition_3gram": agg.avg_repetition_3gram,
            "sensitivity": agg.sensitivity,
            "separate_metrics": agg.separate_metrics.to_dict() if agg.separate_metrics else None,
        }

        print(f"  Script Dom: {agg.avg_target_script_ratio:.1%}")
        print(f"  Semantic:   {agg.avg_semantic_similarity:.3f}" if agg.avg_semantic_similarity else "  Semantic:   N/A")
        print(f"  Degraded:   {agg.degradation_rate:.1%}")

    # 6. Statistical Comparisons
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS")
    print("=" * 60)

    stats_results = run_statistical_comparisons(per_prompt_data)
    results["statistical_comparisons"] = stats_results

    # Print significant findings
    significant = [c for c in stats_results["comparisons"] if c.get("significant", False)]
    if significant:
        print(f"\nSignificant differences (α=0.05, Holm-corrected):")
        for comp in significant:
            print(f"  {comp['schedule_1']} vs {comp['schedule_2']} ({comp['metric']}): "
                  f"p={comp['adjusted_p']:.4f}, effect={comp['effect_size']:.3f}")
    else:
        print("\nNo statistically significant differences found after correction.")

    # 7. Key comparisons summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    baseline = results["schedules"].get("no_steering", {})
    constant = results["schedules"].get("constant", {})

    if baseline and constant:
        script_lift = constant.get("script_dominance", 0) - baseline.get("script_dominance", 0)
        sem_delta = (constant.get("semantic_similarity") or 0) - (baseline.get("semantic_similarity") or 0)
        print(f"\nSteering Effect (constant vs no_steering):")
        print(f"  Script dominance lift: {script_lift:+.1%}")
        print(f"  Semantic similarity Δ: {sem_delta:+.3f}")

    # ------------------------------------------------------------------
    # Publication decision rule (pre-registered style)
    # ------------------------------------------------------------------
    # Primary endpoint: script_ratio vs constant.
    # Adopt an alternative schedule only if:
    #   - Holm-adjusted p < 0.05 AND |effect size| >= 0.2, AND
    #   - it improves script_ratio over constant, AND
    #   - it does not significantly *decrease* semantic_sim vs constant.
    comps = stats_results.get("comparisons", []) or []
    alpha = 0.05
    min_effect = 0.2

    def _pair_comp(metric: str, a: str, b: str) -> dict | None:
        for c in comps:
            if c.get("metric") != metric:
                continue
            s1 = c.get("schedule_1")
            s2 = c.get("schedule_2")
            if {s1, s2} == {a, b}:
                return c
        return None

    def _delta_for(comp: dict, a: str, b: str) -> float | None:
        """Return median(a) - median(b) from a comparison record."""
        md = comp.get("median_diff", None)
        if md is None:
            return None
        if comp.get("schedule_1") == a and comp.get("schedule_2") == b:
            return float(md)
        if comp.get("schedule_1") == b and comp.get("schedule_2") == a:
            return -float(md)
        return None

    candidates = [c["name"] for c in schedules if c["name"] not in ("no_steering", "constant")]
    decision = {
        "primary_metric": "script_ratio",
        "alpha": alpha,
        "min_effect_size": min_effect,
        "candidates": [],
        "chosen_schedule": "constant",
    }

    best = None
    for cand in candidates:
        comp_script = _pair_comp("script_ratio", cand, "constant")
        if not comp_script:
            continue
        adj_p = float(comp_script.get("adjusted_p", 1.0))
        eff = float(comp_script.get("effect_size", 0.0))
        delta_script = _delta_for(comp_script, cand, "constant")
        passes_primary = (
            delta_script is not None
            and delta_script > 0
            and adj_p < alpha
            and abs(eff) >= min_effect
        )

        # Secondary guard: reject schedules that significantly harm semantic similarity.
        comp_sem = _pair_comp("semantic_sim", cand, "constant")
        sem_guard_ok = True
        sem_adj_p = None
        sem_eff = None
        sem_delta = None
        if comp_sem:
            sem_adj_p = float(comp_sem.get("adjusted_p", 1.0))
            sem_eff = float(comp_sem.get("effect_size", 0.0))
            sem_delta = _delta_for(comp_sem, cand, "constant")
            if sem_delta is not None and sem_delta < 0 and sem_adj_p < alpha and abs(sem_eff) >= min_effect:
                sem_guard_ok = False

        decision["candidates"].append(
            {
                "schedule": cand,
                "script_ratio_delta_median": delta_script,
                "script_ratio_adjusted_p": adj_p,
                "script_ratio_effect_size": eff,
                "passes_primary": passes_primary,
                "semantic_sim_delta_median": sem_delta,
                "semantic_sim_adjusted_p": sem_adj_p,
                "semantic_sim_effect_size": sem_eff,
                "passes_semantic_guard": sem_guard_ok,
            }
        )

        if not passes_primary or not sem_guard_ok:
            continue

        if best is None or (delta_script is not None and delta_script > best["delta_script"]):
            best = {"schedule": cand, "delta_script": float(delta_script)}

    if best is not None:
        decision["chosen_schedule"] = best["schedule"]

    results["schedule_recommendation"] = decision
    print(f"\nChosen schedule per decision rule: {decision['chosen_schedule']}")

    # 8. Save
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp15_steering_schedule{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {out_path}")

if __name__ == "__main__":
    main()
