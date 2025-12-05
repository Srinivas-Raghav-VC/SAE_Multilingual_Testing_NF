"""Summarize all experiment JSON results into a text report.

This script reads the JSON files in the `results/` directory and writes a
human-readable summary to `results/summary_report.txt`. The goal is to give
you a single file you can open when writing the paper, with the key numbers
and tables already organized by experiment, language, method, and layer.

Usage:
    python summarize_results.py

Assumes that the experiments (exp1–exp12) have been run and their JSON
outputs exist in `results/`.
"""

import json
from pathlib import Path
from typing import Dict, Any


RESULTS_DIR = Path("results")
REPORT_PATH = RESULTS_DIR / "summary_report.txt"


def _load_json(name: str) -> Any:
    """Load a JSON file from results/ if it exists."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _write_header(out, title: str) -> None:
    out.write("\n" + "=" * 80 + "\n")
    out.write(title + "\n")
    out.write("=" * 80 + "\n\n")


def summarize_exp1(out, data: Dict[str, Any]) -> None:
    """Feature discovery summary: number of language-specific features."""
    if not data:
        out.write("exp1_feature_discovery.json not found.\n")
        return

    _write_header(out, "Experiment 1 – Feature Discovery (Monolinguality)")
    out.write("For each layer and language: number of features with M > 3.0.\n\n")
    out.write(f"{'Layer':<8} {'Lang':<6} {'#features(M>3)':>14} {'#dead':>8}\n")
    out.write("-" * 50 + "\n")

    for layer_str in sorted(data.keys(), key=int):
        layer_data = data[layer_str]
        lang_feats = layer_data.get("lang_features", {})
        dead = layer_data.get("dead_features", {})
        for lang, feats in sorted(lang_feats.items()):
            n = len(feats)
            d = dead.get(lang, 0)
            out.write(f"{layer_str:<8} {lang:<6} {n:>14} {d:>8}\n")


def summarize_exp3(out, data: Dict[str, Any]) -> None:
    """Hindi–Urdu overlap summary."""
    if not data:
        out.write("exp3_hindi_urdu_fixed.json not found.\n")
        return

    _write_header(out, "Experiment 3 – Hindi–Urdu Overlap (Correct Jaccard)")

    layers = data.get("layers", {})
    out.write("Per-layer Jaccard overlaps and script/semantic ratios:\n\n")
    out.write(
        f"{'Layer':<8}"
        f"{'J(HI,UR)%':>12} {'J(HI,EN)%':>12} "
        f"{'Semantic%':>12} {'Script%':>10}\n"
    )
    out.write("-" * 60 + "\n")

    for layer_str in sorted(layers.keys(), key=int):
        ld = layers[layer_str]
        j = ld.get("jaccard_overlaps", {})
        s = ld.get("script_semantic", {})
        out.write(
            f"{layer_str:<8}"
            f"{j.get('hindi_urdu', 0)*100:>12.1f}"
            f"{j.get('hindi_english', 0)*100:>12.1f}"
            f"{s.get('semantic_ratio', 0)*100:>12.1f}"
            f"{s.get('script_ratio', 0)*100:>10.1f}\n"
        )

    tests = data.get("hypothesis_tests", {})
    out.write("\nHypothesis tests:\n")
    for k, v in tests.items():
        out.write(f"  {k}: {'PASS' if v else 'FAIL'}\n")


def summarize_exp5(out, data: Dict[str, Any]) -> None:
    """Hierarchical representation summary."""
    if not data:
        out.write("exp5_hierarchical_analysis.json not found.\n")
        return

    _write_header(out, "Experiment 5 – Hierarchical Language Representation")

    layers = data.get("layer_analyses", [])
    out.write("Per-layer shared and Indic-specific feature counts:\n\n")
    out.write(f"{'Layer':<8} {'#shared':>10} {'#Indic-only':>14}\n")
    out.write("-" * 40 + "\n")
    for entry in sorted(layers, key=lambda e: e.get("layer", 0)):
        layer = entry.get("layer", 0)
        shared = entry.get("n_shared", 0)
        indic = entry.get("n_indic", 0)
        out.write(f"{layer:<8} {shared:>10} {indic:>14}\n")

    hs = data.get("hierarchy_summary", {})
    if hs:
        out.write("\nHierarchy summary (early vs mid vs late):\n")
        for band in ["early_layers", "mid_layers", "late_layers"]:
            b = hs.get(band, {})
            out.write(
                f"  {band}: layers={b.get('range', [])}, "
                f"avg_shared={b.get('avg_shared_features', 0):.1f}, "
                f"avg_hi_ur_overlap={b.get('avg_hi_ur_overlap', 0)*100:.1f}%\n"
            )


def summarize_exp6(out, data: Dict[str, Any]) -> None:
    """Script vs semantics controls summary."""
    if not data:
        out.write("exp6_script_semantics_controls.json not found.\n")
        return

    _write_header(out, "Experiment 6 – Script vs Semantic Controls")
    out.write(
        "Counts of Hindi script-only, Hindi semantic, and family-semantic features per layer.\n\n"
    )
    out.write(
        f"{'Layer':<8} {'HI_script':>10} {'HI_sem':>10} {'Family_sem':>12}\n"
    )
    out.write("-" * 50 + "\n")

    for layer_str in sorted(data.keys(), key=int):
        ld = data[layer_str]
        out.write(
            f"{layer_str:<8}"
            f"{ld.get('hi_script_only', 0):>10}"
            f"{ld.get('hi_semantic', 0):>10}"
            f"{ld.get('family_semantic', 0):>12}\n"
        )


def summarize_exp8(out, data: Dict[str, Any]) -> None:
    """Scaling to 9B and low-resource languages summary."""
    if not data:
        out.write("exp8_scaling_9b_low_resource.json not found.\n")
        return

    _write_header(out, "Experiment 8 – Scaling to 9B and Low-Resource Languages")

    for model_name, mdata in data.items():
        out.write(f"\nModel: {model_name}\n")
        out.write("  Feature coverage (#active features per language and layer):\n")
        out.write(f"  {'Layer':<8} {'Lang':<6} {'#features':>10}\n")
        out.write("  " + "-" * 30 + "\n")
        fc = mdata.get("feature_counts", {})
        for layer_str in sorted(fc.keys(), key=int):
            for lang, cnt in sorted(fc[layer_str].items()):
                out.write(f"  {layer_str:<8} {lang:<6} {cnt:>10}\n")

        out.write("\n  EN→HI steering summary (success_rate, degradation):\n")
        out.write(f"  {'Layer':<8} {'Method':<18} {'Success':>10} {'Degrade':>10}\n")
        out.write("  " + "-" * 50 + "\n")
        steer = mdata.get("steering", {})
        for layer_str in sorted(steer.keys(), key=int):
            for method, md in steer[layer_str].items():
                out.write(
                    f"  {layer_str:<8} {method:<18}"
                    f"{md.get('success_rate', 0)*100:>9.1f}%"
                    f"{md.get('degradation_rate', 0)*100:>9.1f}%\n"
                )


def summarize_exp9(out, data: Dict[str, Any]) -> None:
    """Layer-wise steering sweep summary."""
    if not data:
        out.write("exp9_layer_sweep_steering.json not found.\n")
        return

    _write_header(out, "Experiment 9 – Layer-wise Steering Sweep (per-language best)")

    out.write(
        "For each language and method, we report the best script+semantic success\n"
        "across all layers and strengths (if semantic data missing, script-only).\n\n"
    )
    out.write(
        f"{'Lang':<6} {'Method':<20} {'BestSucc%':>10} "
        f"{'Layer':>8} {'Strength':>10}\n"
    )
    out.write("-" * 60 + "\n")

    for lang, ldata in sorted(data.items()):
        layers = ldata.get("layers", {})
        for layer_str, methods in layers.items():
            layer = int(layer_str)
            for method, mdata in methods.items():
                for strength_str, metrics in mdata.items():
                    strength = float(strength_str)
                    succ = metrics.get("success_rate_script_semantic", None)
                    if succ is None:
                        succ = metrics.get("success_rate_script", 0.0)
                    metrics["__succ"] = succ

        # After annotating, scan again for best per method
        method_best = {}
        for layer_str, methods in layers.items():
            layer = int(layer_str)
            for method, mdata in methods.items():
                for strength_str, metrics in mdata.items():
                    succ = metrics.get("__succ", 0.0)
                    strength = float(strength_str)
                    prev = method_best.get(method, (0.0, None, None))
                    if succ > prev[0]:
                        method_best[method] = (succ, layer, strength)

        for method, (succ, layer, strength) in sorted(method_best.items()):
            out.write(
                f"{lang:<6} {method:<20} "
                f"{succ*100:>9.1f}% {layer:>8} {strength:>10.2f}\n"
            )


def summarize_exp10(out, data: Dict[str, Any]) -> None:
    """Attribution vs activation_diff vs dense summary."""
    if not data:
        out.write("exp10_attribution_occlusion.json not found.\n")
        return

    _write_header(out, "Experiment 10 – Occlusion-Based Attribution Steering")

    layer = data.get("layer", None)
    if layer is not None:
        out.write(f"Layer used for attribution steering: {layer}\n\n")

    steering = data.get("steering_results", {})
    out.write(
        f"{'Method':<24} {'Succ_script%':>14} "
        f"{'Succ_sem%':>12} {'Degrade%':>10}\n"
    )
    out.write("-" * 60 + "\n")
    for method, md in steering.items():
        out.write(
            f"{method:<24}"
            f"{md.get('success_rate_script', 0)*100:>13.1f}%"
            f"{(md.get('success_rate_script_semantic') or 0)*100:>11.1f}%"
            f"{md.get('degradation_rate', 0)*100:>9.1f}%\n"
        )


def summarize_exp11(out, data: Dict[str, Any]) -> None:
    """Calibrated Gemini judge summary."""
    if not data:
        out.write("exp11_judge_calibration.json not found.\n")
        return

    _write_header(out, "Experiment 11 – Calibrated LLM-as-Judge (Gemini)")

    out.write(
        f"{'Lang':<6} {'RawAcc%':>10} {'Theta^%':>10} "
        f"{'CI_low%':>10} {'CI_high%':>10}\n"
    )
    out.write("-" * 60 + "\n")

    for lang, ld in sorted(data.items()):
        out.write(
            f"{lang:<6}"
            f"{ld.get('raw_accuracy', 0)*100:>9.1f}%"
            f"{ld.get('corrected_accuracy', 0)*100:>9.1f}%"
            f"{ld.get('ci_low', 0)*100:>9.1f}%"
            f"{ld.get('ci_high', 0)*100:>9.1f}%\n"
        )


def summarize_exp12(out, data: Dict[str, Any]) -> None:
    """QA degradation under steering summary."""
    if not data:
        out.write("exp12_qa_degradation.json not found.\n")
        return

    _write_header(out, "Experiment 12 – QA Degradation Under Steering (MLQA + IndicQA)")

    out.write(
        f"{'Task':<14} {'Lang':<6} {'Mode':<10} "
        f"{'Succ_script%':>14} {'Succ_sem%':>12} "
        f"{'SemSim':>8} {'Degrade%':>10}\n"
    )
    out.write("-" * 90 + "\n")

    for key, kd in sorted(data.items()):
        # key looks like "mlqa_hi" or "indicqa_bn"
        if "_" in key:
            task, lang = key.split("_", 1)
        else:
            task, lang = key, "??"

        for mode in ["baseline", "steered"]:
            md = kd.get(mode, {})
            out.write(
                f"{task:<14} {lang:<6} {mode:<10}"
                f"{(md.get('success_rate_script') or 0)*100:>13.1f}%"
                f"{(md.get('success_rate_script_semantic') or 0)*100:>11.1f}%"
                f"{(md.get('avg_semantic_similarity') or 0):>8.2f}"
                f"{(md.get('degradation_rate') or 0)*100:>9.1f}%\n"
            )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(REPORT_PATH, "w") as out:
        _write_header(out, "SAE Multilingual Steering – Summary Report")
        out.write("This report summarizes key metrics from all experiments.\n")
        out.write("All numbers are taken directly from JSON files in results/.\n")

        # Load JSONs once
        exp1 = _load_json("exp1_feature_discovery")
        exp3 = _load_json("exp3_hindi_urdu_fixed")
        exp5 = _load_json("exp5_hierarchical_analysis")
        exp6 = _load_json("exp6_script_semantics_controls")
        exp8 = _load_json("exp8_scaling_9b_low_resource")
        exp9 = _load_json("exp9_layer_sweep_steering")
        exp10 = _load_json("exp10_attribution_occlusion")
        exp11 = _load_json("exp11_judge_calibration")
        exp12 = _load_json("exp12_qa_degradation")

        summarize_exp1(out, exp1 or {})
        summarize_exp3(out, exp3 or {})
        summarize_exp5(out, exp5 or {})
        summarize_exp6(out, exp6 or {})
        summarize_exp8(out, exp8 or {})
        summarize_exp9(out, exp9 or {})
        summarize_exp10(out, exp10 or {})
        summarize_exp11(out, exp11 or {})
        summarize_exp12(out, exp12 or {})

    print(f"Summary report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

