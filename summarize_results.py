"""Summarize all experiment JSON results into a text report.

This script reads the JSON files in the `results/` directory and writes a
human-readable summary to `results/summary_report.txt`. The goal is to give
you a single file you can open when writing the paper, with the key numbers
and tables already organized by experiment, language, method, and layer.

Usage:
    python summarize_results.py

It will summarize all available experiments (exp1–exp16). If some JSON files
are missing (because a given experiment did not run), it will note that and
continue.
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


def summarize_exp2(out, data: Dict[str, Any]) -> None:
    """Steering method comparison (simple EN→HI sanity check)."""
    if not data:
        out.write("exp2_steering_comparison.json not found.\n")
        return

    _write_header(out, "Experiment 2 – Steering Method Comparison (EN→HI)")
    out.write(
        "Fast sanity check comparing dense / activation_diff / monolinguality / random\n"
        "for EN→HI at a few layers. Use this to understand method behavior before\n"
        "looking at the full multi-language sweep in Exp9.\n\n"
    )

    layers = data.get("layers", {})
    if not layers:
        out.write("No 'layers' key in exp2 results.\n")
        return

    # Focus on layer 13 if available, else the first layer
    layer_key = "13" if "13" in layers else sorted(layers.keys(), key=int)[0]
    methods = layers[layer_key].get("methods", {})

    out.write(f"Layer: {layer_key}\n\n")
    out.write(
        f"{'Method':<20} {'BestSucc%':>10} {'BestStrength':>12}\n"
        + "-" * 45 + "\n"
    )

    for method, mdata in methods.items():
        best_strength = None
        best_rate = -1.0
        for s_str, metrics in mdata.items():
            rate = metrics.get("success_rate", 0.0)
            s_val = float(s_str)
            if rate > best_rate:
                best_rate = rate
                best_strength = s_val
        out.write(
            f"{method:<20} {best_rate*100:>9.1f}% {best_strength:>12.2f}\n"
        )

    out.write(
        "\nNote: Exp2 uses FLORES-only EN/HI and no LLM judge; it is a simple,\n"
        "low-cost comparison. For full multi-language results, see Exp9 below.\n"
    )


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

    out.write(
        "\nNote: These categories are defined via activation thresholds and\n"
        "transliteration/noise controls; see Exp13 for group ablations that\n"
        "test their causal impact on script vs semantics.\n"
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

    out.write(
        "\nNote: If semantic_success_rate is missing in the JSON for a given\n"
        "method/strength, the best score above is based on script-only success.\n"
        "Check exp9_layer_sweep_steering.json for full metric details.\n"
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

    # JSON layout from exp11_judge_calibration.py:
    # {
    #   "layer": ...,
    #   "strength": ...,
    #   "languages": {
    #       "hi": {"raw_accuracy": ..., "corrected_accuracy": ..., ...},
    #       "de": {...}
    #   }
    # }
    langs_block = data.get("languages", {})
    for lang, ld in sorted(langs_block.items()):
        out.write(
            f"{lang:<6}"
            f"{ld.get('raw_accuracy', 0)*100:>9.1f}%"
            f"{ld.get('corrected_accuracy', 0)*100:>9.1f}%"
            f"{ld.get('ci_low', 0)*100:>9.1f}%"
            f"{ld.get('ci_high', 0)*100:>9.1f}%\n"
        )

    out.write(
        "\nNote: Exp11 currently calibrates the Gemini judge only for the\n"
        "languages listed above (e.g., hi, de). When interpreting LLM-judge\n"
        "metrics for other languages, treat them as uncalibrated.\n"
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

    out.write(
        "\nNote: Tasks with very low baseline script/semantic success may not\n"
        "be meaningful to steer. Inspect exp12_qa_degradation.json to see\n"
        "per-task baselines before drawing strong conclusions.\n"
    )


def summarize_exp4(out, data: Dict[str, Any]) -> None:
    """Spillover analysis summary."""
    if not data:
        out.write("exp4_spillover.json not found.\n")
        return

    _write_header(out, "Experiment 4 – Language Steering Spillover (EN→HI vs EN→DE)")

    # Handle both old format ("layers") and new format ("en_to_hi_layers", "en_to_de_layers")
    layers_hi = data.get("en_to_hi_layers", data.get("layers", {}))
    layers_de = data.get("en_to_de_layers", {})
    tests = data.get("hypothesis_tests", {})
    family_hi = data.get("family_analysis_en_to_hi", data.get("family_analysis", {}))
    family_de = data.get("family_analysis_en_to_de", {})

    out.write("Hypothesis tests (using late layer, EN→HI):\n")
    for k, v in tests.items():
        out.write(f"  {k}: {'PASS' if v else 'FAIL'}\n")

    # EN→HI spillover matrix
    if layers_hi:
        out.write("\nEN→HI spillover matrix by layer and strength (hi/ur/bn/en/de):\n")
        out.write(f"{'Layer':<8} {'Strength':<10} {'hi%':>8} {'ur%':>8} {'bn%':>8} {'en%':>8} {'de%':>8}\n")
        out.write("-" * 60 + "\n")
        for layer_str in sorted(layers_hi.keys(), key=lambda x: int(x)):
            for strength_str, res in sorted(layers_hi[layer_str].items(), key=lambda x: float(x[0])):
                dist = res.get("language_distribution", {})
                out.write(
                    f"{layer_str:<8} {strength_str:<10}"
                    f"{dist.get('hi', 0):>8.1f}{dist.get('ur', 0):>8.1f}"
                    f"{dist.get('bn', 0):>8.1f}{dist.get('en', 0):>8.1f}"
                    f"{dist.get('de', 0):>8.1f}\n"
                )

    # EN→DE spillover matrix (control)
    if layers_de:
        out.write("\nEN→DE (control) spillover matrix by layer and strength:\n")
        out.write(f"{'Layer':<8} {'Strength':<10} {'hi%':>8} {'ur%':>8} {'bn%':>8} {'en%':>8} {'de%':>8}\n")
        out.write("-" * 60 + "\n")
        for layer_str in sorted(layers_de.keys(), key=lambda x: int(x)):
            for strength_str, res in sorted(layers_de[layer_str].items(), key=lambda x: float(x[0])):
                dist = res.get("language_distribution", {})
                out.write(
                    f"{layer_str:<8} {strength_str:<10}"
                    f"{dist.get('hi', 0):>8.1f}{dist.get('ur', 0):>8.1f}"
                    f"{dist.get('bn', 0):>8.1f}{dist.get('en', 0):>8.1f}"
                    f"{dist.get('de', 0):>8.1f}\n"
                )

    out.write("\nFamily-level spillover (Indo-Aryan vs others) by strength:\n")
    if family_hi:
        out.write("  EN→HI:\n")
        for strength, fams in family_hi.items():
            out.write(f"    strength={strength}: {fams}\n")
    if family_de:
        out.write("  EN→DE (control):\n")
        for strength, fams in family_de.items():
            out.write(f"    strength={strength}: {fams}\n")


def summarize_exp7(out, data: Any) -> None:
    """Causal probing of individual features summary."""
    if not data:
        out.write("exp7_causal_feature_probing.json not found.\n")
        return

    _write_header(out, "Experiment 7 – Causal Probing of Individual SAE Features")
    out.write(
        "Each entry is a (layer, feature, direction, strength) with deltas in\n"
        "script, semantics, and degradation. We summarize the top features by\n"
        "absolute delta_script and absolute delta_semantic.\n\n"
    )

    # data is a list of dicts
    # Select top-k by |delta_script| and |delta_semantic|
    top_k = 10
    by_script = sorted(
        data,
        key=lambda e: abs(e.get("delta_script", 0.0)),
        reverse=True,
    )[:top_k]
    by_sem = sorted(
        data,
        key=lambda e: abs(e.get("delta_semantic", 0.0)),
        reverse=True,
    )[:top_k]

    out.write("Top features by |Δscript|:\n")
    out.write(f"{'Layer':<8} {'Feat':<8} {'dir':<4} {'str':>4} {'Δscript':>10} {'Δsem':>10} {'Δdeg':>10}\n")
    out.write("-" * 70 + "\n")
    for e in by_script:
        out.write(
            f"{e.get('layer', 0):<8} {e.get('feature_idx', 0):<8} "
            f"{e.get('method', ''):<4} {e.get('strength', 0):>4.1f} "
            f"{e.get('delta_script', 0):>10.3f}{e.get('delta_semantic', 0):>10.3f}"
            f"{e.get('delta_degradation', 0):>10.3f}\n"
        )

    out.write("\nTop features by |Δsemantic|:\n")
    out.write(f"{'Layer':<8} {'Feat':<8} {'dir':<4} {'str':>4} {'Δscript':>10} {'Δsem':>10} {'Δdeg':>10}\n")
    out.write("-" * 70 + "\n")
    for e in by_sem:
        out.write(
            f"{e.get('layer', 0):<8} {e.get('feature_idx', 0):<8} "
            f"{e.get('method', ''):<4} {e.get('strength', 0):>4.1f} "
            f"{e.get('delta_script', 0):>10.3f}{e.get('delta_semantic', 0):>10.3f}"
            f"{e.get('delta_degradation', 0):>10.3f}\n"
        )


def summarize_exp13(out, data: Dict[str, Any]) -> None:
    """Group ablations for script vs semantic feature groups."""
    if not data:
        out.write("exp13_script_semantic_ablation.json not found.\n")
        return

    _write_header(out, "Experiment 13 – Group Ablation (Script vs Semantic Features)")

    layer = data.get("layer", None)
    if layer is not None:
        out.write(f"Layer used: {layer}\n")

    out.write(
        f"#HI script-only features: {data.get('n_hi_script_only', 0)}\n"
        f"#HI semantic features:    {data.get('n_hi_semantic', 0)}\n\n"
    )

    for group_name in ["hi_script_only_ablation", "hi_semantic_ablation"]:
        res = data.get(group_name, {})
        if not res:
            continue
        out.write(f"{group_name}:\n")
        out.write(
            f"  Δscript_ratio: {res.get('delta_script_ratio', 0):+.4f}\n"
            f"  Δsemantic_sim: {res.get('delta_semantic_similarity', 0):+.4f}\n"
            f"  Δdegradation:  {res.get('delta_degradation_rate', 0):+.4f}\n"
        )

    out.write(
        "\nInterpretation hint: If script-only ablation mainly shifts script_ratio\n"
        "with small semantic change, and semantic-group ablation mainly hurts\n"
        "semantic similarity, this supports the script-sensitive vs invariant\n"
        "interpretation of these feature groups.\n"
    )


def summarize_exp14(out, data: Dict[str, Any]) -> None:
    """Layer-wise cross-lingual alignment summary."""
    if not data:
        out.write("exp14_language_agnostic_space.json not found.\n")
        return

    _write_header(out, "Experiment 14 – Cross-Lingual Alignment (Language-Agnostic Space)")

    langs = data.get("languages", [])
    layers = data.get("layers", {})

    out.write("Average cosine similarity between sentence embeddings (FLORES):\n\n")
    out.write(f"{'Layer':<8} {'en-hi':>10} {'en-ur':>10} {'en-bn':>10} {'en-ta':>10} {'en-te':>10}\n")
    out.write("-" * 60 + "\n")

    best_layer = None
    best_en_hi = -1.0

    for layer_str in sorted(layers.keys(), key=int):
        pair = layers[layer_str].get("pairwise_cosine", {})
        en_hi = pair.get("en-hi", 0.0)
        en_ur = pair.get("en-ur", 0.0)
        en_bn = pair.get("en-bn", 0.0)
        en_ta = pair.get("en-ta", 0.0)
        en_te = pair.get("en-te", 0.0)
        out.write(
            f"{layer_str:<8}{en_hi:>10.3f}{en_ur:>10.3f}{en_bn:>10.3f}{en_ta:>10.3f}{en_te:>10.3f}\n"
        )
        if en_hi > best_en_hi:
            best_en_hi = en_hi
            best_layer = int(layer_str)

    out.write(
        f"\nLayer with highest EN–HI cosine (approx. most aligned): {best_layer} "
        f"(cos≈{best_en_hi:.3f}). If this lies in mid-layers, it supports a\n"
        "partially language-agnostic region for EN+Indic.\n"
    )


def summarize_exp15(out, data: Dict[str, Any]) -> None:
    """Directional symmetry of steering (EN→Indic vs Indic→EN)."""
    if not data:
        out.write("exp15_directional_symmetry.json not found.\n")
        return

    _write_header(out, "Experiment 15 – Directional Symmetry of Steering (EN↔Indic)")

    out.write(
        f"{'Lang':<6} {'Dir':<8} {'ΔSucc_script%':>14} {'ΔDegrade%':>12}\n"
        + "-" * 50 + "\n"
    )

    for lang, ld in sorted(data.items()):
        en_to_l = ld.get("en_to_l", {})
        l_to_en = ld.get("l_to_en", {})
        out.write(
            f"{lang:<6} {'EN→L':<8}"
            f"{en_to_l.get('delta_success_script', 0)*100:>13.1f}%"
            f"{en_to_l.get('delta_degradation_rate', 0)*100:>11.1f}%\n"
        )
        out.write(
            f"{lang:<6} {'L→EN':<8}"
            f"{l_to_en.get('delta_success_script', 0)*100:>13.1f}%"
            f"{l_to_en.get('delta_degradation_rate', 0)*100:>11.1f}%\n"
        )

    out.write(
        "\nIf EN→Indic consistently shows higher Δsuccess than Indic→EN at the\n"
        "same layer, this suggests an English-anchored representation. If they\n"
        "are similar, steering may be more symmetric.\n"
    )


def summarize_exp16(out, data: Dict[str, Any]) -> None:
    """Code-mix and noise robustness summary."""
    if not data:
        out.write("exp16_code_mix_robustness.json not found.\n")
        return

    _write_header(out, "Experiment 16 – Code-Mix and Noise Robustness (EN→HI)")

    out.write(
        f"{'Condition':<14} {'ΔSucc_script%':>14} {'ΔDegrade%':>12}\n"
        + "-" * 50 + "\n"
    )
    for cond in ["clean_en", "en_plus_deva", "hinglish_mix"]:
        md = data.get(cond, {})
        out.write(
            f"{cond:<14}"
            f"{md.get('delta_success_script', 0)*100:>13.1f}%"
            f"{md.get('delta_degradation_rate', 0)*100:>11.1f}%\n"
        )

    out.write(
        "\nIf steering effectiveness drops substantially for code-mixed/noisy\n"
        "prompts compared to clean English, the steering direction may be\n"
        "tightly coupled to well-formed inputs rather than robust semantics.\n"
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(REPORT_PATH, "w") as out:
        _write_header(out, "SAE Multilingual Steering – Summary Report")
        out.write("This report summarizes key metrics from all experiments.\n")
        out.write("All numbers are taken directly from JSON files in results/.\n")

        # Load JSONs once
        exp1 = _load_json("exp1_feature_discovery")
        exp2 = _load_json("exp2_steering_comparison")
        exp3 = _load_json("exp3_hindi_urdu_fixed")
        exp5 = _load_json("exp5_hierarchical_analysis")
        exp6 = _load_json("exp6_script_semantics_controls")
        exp4 = _load_json("exp4_spillover")
        exp8 = _load_json("exp8_scaling_9b_low_resource")
        exp9 = _load_json("exp9_layer_sweep_steering")
        exp10 = _load_json("exp10_attribution_occlusion")
        exp11 = _load_json("exp11_judge_calibration")
        exp12 = _load_json("exp12_qa_degradation")
        exp7 = _load_json("exp7_causal_feature_probing")
        exp13 = _load_json("exp13_script_semantic_ablation")
        exp14 = _load_json("exp14_language_agnostic_space")
        exp15 = _load_json("exp15_directional_symmetry")
        exp16 = _load_json("exp16_code_mix_robustness")

        summarize_exp1(out, exp1 or {})
        summarize_exp2(out, exp2 or {})
        summarize_exp3(out, exp3 or {})
        summarize_exp4(out, exp4 or {})
        summarize_exp5(out, exp5 or {})
        summarize_exp6(out, exp6 or {})
        summarize_exp8(out, exp8 or {})
        summarize_exp9(out, exp9 or {})
        summarize_exp10(out, exp10 or {})
        summarize_exp11(out, exp11 or {})
        summarize_exp12(out, exp12 or {})
        summarize_exp7(out, exp7 or [])
        summarize_exp13(out, exp13 or {})
        summarize_exp14(out, exp14 or {})
        summarize_exp15(out, exp15 or {})
        summarize_exp16(out, exp16 or {})

    print(f"Summary report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
