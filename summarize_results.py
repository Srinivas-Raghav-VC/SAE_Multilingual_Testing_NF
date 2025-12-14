"""Summarize all experiment JSON results into a text report.

This script reads the JSON files in the `results/` directory and writes a
human-readable summary to `results/summary_report.txt`. The goal is to give
you a single file you can open when writing the paper, with the key numbers
and tables already organized by experiment, language, method, and layer.

Usage:
    python summarize_results.py

It will summarize all available experiments used in the publication run
(Exp1/3/5/8/9/10/11/12/13/14/15/18/19/20/21/22/23/24/25).
If some JSON files are missing (because a given experiment did not run), it
will note that and continue.
"""

import json
from pathlib import Path
from typing import Dict, Any


RESULTS_DIR = Path("results")
REPORT_PATH = RESULTS_DIR / "summary_report.txt"
TABLES_DIR = RESULTS_DIR / "tables"


def _load_json(name: str) -> Any:
    """Load a JSON file from results/ if it exists."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_json_variants(name: str) -> Dict[str, Any]:
    """Load 2B + 9B variants of an experiment if available.

    Returns mapping like {"2b": data2b, "9b": data9b}.
    """
    variants: Dict[str, Any] = {}
    base = _load_json(name)
    if base is not None:
        variants["2b"] = base
    v9 = _load_json(f"{name}_9b")
    if v9 is not None:
        variants["9b"] = v9
    return variants


def _write_header(out, title: str) -> None:
    out.write("\n" + "=" * 80 + "\n")
    out.write(title + "\n")
    out.write("=" * 80 + "\n\n")


def _write_latex_table(path: Path, latex: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def write_table_exp1_feature_discovery(data: Dict[str, Any], suffix: str = "") -> None:
    """Write paper-ready LaTeX table for Exp1 into results/tables/."""
    if not data:
        return

    layers = sorted([int(k) for k in data.keys() if str(k).isdigit()])
    if not layers:
        return

    lang_set = set()
    for layer in layers:
        layer_data = data.get(str(layer), {})
        lang_set.update((layer_data.get("lang_features") or {}).keys())

    order = ["hi", "bn", "ta", "te", "en", "ur", "de", "ar"]
    langs = [l for l in order if l in lang_set] + sorted([l for l in lang_set if l not in order])
    if not langs:
        return

    label = "tab:feature-discovery" + (f"-{suffix.strip('_')}" if suffix else "")
    cols = "@{}l" + ("r" * len(langs)) + "r@{}"

    header = " & ".join([f"\\textbf{{{l.upper()}}}" for l in langs])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Language-specific detector features per layer} (monolinguality $M>3$). Each cell is the count of SAE features strongly selective for that language.}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        f"\\textbf{{Layer}} & {header} & \\textbf{{Total}} \\\\",
        "\\midrule",
    ]

    for layer in layers:
        layer_data = data.get(str(layer), {})
        lang_feats = layer_data.get("lang_features", {}) or {}
        counts = [len(lang_feats.get(l, []) or []) for l in langs]
        total = sum(counts)
        row = " & ".join([str(layer)] + [str(c) for c in counts] + [str(total)]) + " \\\\"
        lines.append(row)

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    _write_latex_table(TABLES_DIR / f"tab_exp1_feature_discovery{suffix}.tex", "\n".join(lines))


def write_table_exp3_hi_ur_overlap(data: Dict[str, Any], suffix: str = "") -> None:
    """Write paper-ready LaTeX table for Exp3 into results/tables/."""
    if not data:
        return

    layers_data = data.get("layers", {}) or {}
    if not layers_data:
        return

    layers = sorted([int(k) for k in layers_data.keys() if str(k).isdigit()])
    if not layers:
        return

    label = "tab:hindi-urdu-overlap" + (f"-{suffix.strip('_')}" if suffix else "")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Hindi--Urdu overlap and script-specific shell (Exp3).} Jaccard overlaps are computed on active feature sets; script-only features are Hindi-only + Urdu-only.}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "\\textbf{Layer} & \\textbf{HI--UR} & \\textbf{HI--EN} & \\textbf{Semantic} & \\textbf{Script} & \\textbf{Script \\%} \\\\",
        "\\midrule",
    ]

    for layer in layers:
        ld = layers_data.get(str(layer), {}) or {}
        j = ld.get("jaccard_overlaps", {}) or {}
        s = ld.get("script_semantic", {}) or {}

        hi_ur = float(j.get("hindi_urdu", 0.0)) * 100.0
        hi_en = float(j.get("hindi_english", 0.0)) * 100.0
        semantic = int(s.get("semantic", 0) or 0)
        script = int(s.get("hindi_script", 0) or 0) + int(s.get("urdu_script", 0) or 0)
        denom = max(semantic + script, 1)
        script_pct = 100.0 * float(script) / float(denom)

        lines.append(
            f"{layer} & {hi_ur:.1f}\\% & {hi_en:.1f}\\% & {semantic:,} & {script:,} & {script_pct:.1f}\\% \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    _write_latex_table(TABLES_DIR / f"tab_exp3_hi_ur_overlap{suffix}.tex", "\n".join(lines))


def write_table_exp6_script_semantics(data: Dict[str, Any], suffix: str = "") -> None:
    """Write paper-ready LaTeX table for Exp6 into results/tables/."""
    if not data:
        return

    layers = sorted([int(k) for k in data.keys() if str(k).isdigit()])
    if not layers:
        return

    # Require the Jaccard keys to avoid generating a misleading table.
    if not all("jaccard_hi_deva_hi_latin" in (data.get(str(l), {}) or {}) for l in layers):
        return

    label = "tab:script-semantics" + (f"-{suffix.strip('_')}" if suffix else "")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Script vs. semantics control (Exp6).} Transliteration and noise controls separate script-only from script-robust (semantic) features.}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "\\textbf{Layer} & \\textbf{HI--Latin} & \\textbf{HI--Noise} & \\textbf{Semantic} & \\textbf{Script-only} \\\\",
        "\\midrule",
    ]

    for layer in layers:
        d = data.get(str(layer), {}) or {}
        hi_latin = float(d.get("jaccard_hi_deva_hi_latin", 0.0)) * 100.0
        hi_noise = float(d.get("jaccard_hi_deva_noise", 0.0)) * 100.0
        sem = int(d.get("hi_semantic", 0) or 0)
        script_only = int(d.get("hi_script_only", 0) or 0)
        lines.append(f"{layer} & {hi_latin:.1f}\\% & {hi_noise:.1f}\\% & {sem:,} & {script_only:,} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    _write_latex_table(TABLES_DIR / f"tab_exp6_script_semantics{suffix}.tex", "\n".join(lines))


def write_table_exp5_hierarchy(data: Dict[str, Any], suffix: str = "") -> None:
    """Write paper-ready LaTeX table for Exp5 into results/tables/."""
    if not data:
        return

    analyses = data.get("layer_analyses", []) or []
    if not analyses:
        return

    layers = sorted({int(e.get("layer")) for e in analyses if e.get("layer") is not None})
    if not layers:
        return
    max_layer = max(layers)

    def layer_group(layer: int) -> str:
        frac = float(layer) / float(max_layer) if max_layer > 0 else 0.0
        if frac <= 0.33:
            return "Early"
        if frac <= 0.66:
            return "Mid"
        return "Late"

    groups = {"Early": [], "Mid": [], "Late": []}
    for entry in analyses:
        layer = entry.get("layer", None)
        if layer is None:
            continue
        groups[layer_group(int(layer))].append(entry)

    label = "tab:hierarchical" + (f"-{suffix.strip('_')}" if suffix else "")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Hierarchical feature organization (Exp5).} We aggregate per-layer analyses into early/mid/late depth bands and report mean shared/Indic-only counts and mean HI--UR overlap.}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{@{}llrrr@{}}",
        "\\toprule",
        "\\textbf{Stage} & \\textbf{Layers} & \\textbf{Shared} & \\textbf{HI--UR} & \\textbf{Indic-only} \\\\",
        "\\midrule",
    ]

    for stage in ["Early", "Mid", "Late"]:
        entries = groups.get(stage, [])
        if not entries:
            continue

        layer_list = ", ".join(str(int(e.get("layer"))) for e in entries if e.get("layer") is not None)
        shared_vals = [int(e.get("n_shared", 0) or 0) for e in entries]
        indic_vals = [int(e.get("n_indic", 0) or 0) for e in entries]
        hi_ur_vals = []
        for e in entries:
            overlaps = e.get("overlaps", {}) or {}
            v = overlaps.get("hi-ur", overlaps.get("ur-hi", 0.0))
            hi_ur_vals.append(float(v or 0.0))

        avg_shared = int(round(sum(shared_vals) / len(shared_vals))) if shared_vals else 0
        avg_indic = int(round(sum(indic_vals) / len(indic_vals))) if indic_vals else 0
        avg_hi_ur = (sum(hi_ur_vals) / len(hi_ur_vals)) * 100.0 if hi_ur_vals else 0.0

        lines.append(
            f"{stage} & {layer_list} & {avg_shared:,} & {avg_hi_ur:.1f}\\% & {avg_indic:,} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    _write_latex_table(TABLES_DIR / f"tab_exp5_hierarchy{suffix}.tex", "\n".join(lines))


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

    # Metadata (semantic reference + truncation rate), stored per-language in Exp9 JSON.
    any_block = next((v for v in data.values() if isinstance(v, dict)), None)
    if any_block:
        ref_mode = any_block.get("semantic_reference", None)
        trunc = any_block.get("semantic_truncation_stats", None)
        if ref_mode:
            out.write(f"Semantic reference mode: {ref_mode}\n")
        if isinstance(trunc, dict):
            calls = int(trunc.get("calls", 0) or 0)
            truncated = int(trunc.get("truncated_texts", 0) or 0)
            rate = 0.0 if calls <= 0 else (truncated / calls) * 100.0
            out.write(f"LaBSE truncation: {truncated}/{calls} texts ({rate:.1f}%)\n")
        out.write("\n")

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

    ref_mode = data.get("semantic_reference", None)
    trunc = data.get("semantic_truncation_stats", None)
    if ref_mode:
        out.write(f"Semantic reference mode: {ref_mode}\n")
    if isinstance(trunc, dict):
        calls = int(trunc.get("calls", 0) or 0)
        truncated = int(trunc.get("truncated_texts", 0) or 0)
        rate = 0.0 if calls <= 0 else (truncated / calls) * 100.0
        out.write(f"LaBSE truncation: {truncated}/{calls} texts ({rate:.1f}%)\n")
    if ref_mode or isinstance(trunc, dict):
        out.write("\n")

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

    # Truncation metadata (stored per-task).
    any_task = next((v for v in data.values() if isinstance(v, dict)), None)
    if any_task:
        trunc = any_task.get("semantic_truncation_stats", None)
        if isinstance(trunc, dict):
            calls = int(trunc.get("calls", 0) or 0)
            truncated = int(trunc.get("truncated_texts", 0) or 0)
            rate = 0.0 if calls <= 0 else (truncated / calls) * 100.0
            out.write(f"LaBSE truncation: {truncated}/{calls} texts ({rate:.1f}%)\n\n")

    out.write(
        f"{'Task':<14} {'Lang':<6} {'Mode':<10} "
        f"{'Succ_script%':>12} {'Degrade%':>10} "
        f"{'QA_EM':>8} {'QA_F1':>8} "
        f"{'BasePresSem':>12}\n"
    )
    out.write("-" * 92 + "\n")

    for key, kd in sorted(data.items()):
        # key looks like "mlqa_hi" or "indicqa_bn"
        if "_" in key:
            task, lang = key.split("_", 1)
        else:
            task, lang = key, "??"

        for mode in ["baseline", "steered"]:
            md = kd.get(mode, {})
            base_pres = md.get("baseline_preservation_semantic_mean", None)
            base_pres_str = f"{base_pres:>12.2f}" if isinstance(base_pres, (int, float)) else f"{'n/a':>12}"
            out.write(
                f"{task:<14} {lang:<6} {mode:<10}"
                f"{(md.get('success_rate_script') or 0)*100:>11.1f}%"
                f"{(md.get('degradation_rate') or 0)*100:>9.1f}%"
                f"{(md.get('qa_exact_match') or 0):>8.2f}"
                f"{(md.get('qa_f1') or 0):>8.2f}"
                f"{base_pres_str}\n"
            )

        decision = kd.get("decision", {}) or {}
        qa_f1_dec = decision.get("qa_f1", None)
        if isinstance(qa_f1_dec, dict):
            interp = qa_f1_dec.get("interpretation")
            if interp:
                out.write(
                    f"  Decision (QA_F1): {interp} (meanΔ={qa_f1_dec.get('mean_delta')}, "
                    f"CI={qa_f1_dec.get('ci_95_mean_delta')})\n"
                )

    out.write(
        "\nNote: For QA, the most relevant success signals are QA_EM/QA_F1.\n"
        "BasePresSem is a baseline-preservation proxy: LaBSE(baseline, steered).\n"
        "Inspect exp12_qa_degradation.json for per-task details.\n"
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

    seed = data.get("group_ablation_seed", None)
    if seed is not None:
        out.write(f"Random baseline seed: {seed}\n")
    max_feats = data.get("group_ablation_max_features", None)
    if max_feats is not None:
        out.write(f"Max features cap: {max_feats}\n")
    out.write("\n")

    for group_name in [
        "hi_script_only_ablation",
        "hi_semantic_ablation",
        "random_script_only_ablation",
        "random_semantic_ablation",
    ]:
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
    """Steering schedule ablation summary."""
    if not data:
        out.write("exp15_steering_schedule.json not found.\n")
        return

    _write_header(out, "Experiment 15 – Steering Schedule Ablation")

    layer = data.get("layer", None)
    tgt = data.get("target_language", None)
    n_prompts = data.get("n_prompts", None)
    out.write(f"Target: {tgt}, layer: {layer}, n_prompts: {n_prompts}\n\n")

    rec = data.get("schedule_recommendation", {}) or {}
    chosen = rec.get("chosen_schedule", data.get("chosen_schedule", "constant"))
    out.write(f"Chosen schedule (decision rule): {chosen}\n\n")

    schedules = data.get("schedules", {}) or {}
    out.write(
        f"{'Schedule':<18} {'ScriptDom%':>10} {'Succ%':>8} {'Sem':>8} {'Degrade%':>10}\n"
    )
    out.write("-" * 62 + "\n")
    for name, sd in sorted(schedules.items()):
        out.write(
            f"{name:<18}"
            f"{(sd.get('script_dominance') or 0)*100:>9.1f}%"
            f"{(sd.get('success_rate') or 0)*100:>7.1f}%"
            f"{(sd.get('semantic_similarity') or 0):>8.3f}"
            f"{(sd.get('degradation_rate') or 0)*100:>9.1f}%\n"
        )
    out.write("\n")


def summarize_exp18(out, data: Dict[str, Any]) -> None:
    """Typological feature analysis summary."""
    if not data:
        out.write("exp18_typological_features.json not found.\n")
        return

    _write_header(out, "Experiment 18 – Typological Feature Analysis")

    layer = data.get("layer", None)
    out.write(f"Layer: {layer}\n\n")

    var = data.get("variance_analysis", {}) or {}
    out.write(f"Family R²:     {var.get('family_r_squared')}\n")
    out.write(f"Retroflex R²:  {var.get('retroflex_r_squared')}\n")
    out.write(f"N pairs:       {var.get('n_pairs')}\n")
    out.write(f"Winner:        {var.get('interpretation')}\n\n")


def summarize_exp19(out, data: Dict[str, Any]) -> None:
    """Cross-layer causal profiles summary."""
    if not data:
        out.write("exp19_crosslayer_causal.json not found.\n")
        return

    _write_header(out, "Experiment 19 – Cross-Layer Causal Profiles")

    fam = data.get("family_comparison", {}) or {}
    for lang, info in sorted(fam.items()):
        if not isinstance(info, dict):
            continue
        out.write(
            f"{lang.upper()}: peak causal layer={info.get('peak_causal_layer')}, "
            f"peak importance={info.get('peak_causal_importance')}\n"
        )

    cross = data.get("cross_family", {}) or {}
    if cross:
        out.write(
            f"\nCross-family: IA peak={cross.get('indo_aryan_peak_layer')}, "
            f"DR peak={cross.get('dravidian_peak_layer')}, "
            f"same_peak={cross.get('same_peak_layer')}\n"
        )
    out.write("\n")


def summarize_exp20(out, data: Dict[str, Any]) -> None:
    """Training frequency control summary."""
    if not data:
        out.write("exp20_training_freq_control.json not found.\n")
        return

    _write_header(out, "Experiment 20 – Training Frequency Control")

    corr = data.get("correlation_analysis", {}) or {}
    out.write(
        f"Spearman ρ={corr.get('spearman_r')}, p={corr.get('spearman_p')}, "
        f"R²={corr.get('r_squared')}\n"
    )
    out.write(f"Interpretation: {corr.get('interpretation', 'n/a')}\n\n")

    low = data.get("low_resource_analysis", {}) or {}
    out.write(
        f"Malayalam clusters with Dravidian: {low.get('malayalam_clusters_with_dravidian')}\n\n"
    )


def summarize_exp21(out, data: Dict[str, Any]) -> None:
    """Indo-Aryan vs Dravidian separation summary."""
    if not data:
        out.write("exp21_family_separation.json not found.\n")
        return

    _write_header(out, "Experiment 21 – Indo-Aryan vs Dravidian Separation")

    overlap = data.get("family_overlaps", {}) or {}
    out.write(
        f"Separation ratio={overlap.get('separation_ratio')}, "
        f"CI=[{overlap.get('separation_ci_low')}, {overlap.get('separation_ci_high')}], "
        f"p={overlap.get('p_value')}\n"
    )

    transfer = data.get("transfer_summary", {}) or {}
    if transfer:
        gap_test = transfer.get("gap_test", {}) or {}
        out.write(
            f"Transfer gap={transfer.get('transfer_gap')}, "
            f"adj_p={gap_test.get('adjusted_p')}, "
            f"power≈{transfer.get('power_gap')}\n"
        )

    fals = data.get("falsification", {}) or {}
    if fals:
        out.write(
            f"Falsification: unified_supported={fals.get('unified_indic_supported')}, "
            f"distinct_supported={fals.get('distinct_families_supported')}\n"
        )
    out.write("\n")


def summarize_exp22(out, data: Dict[str, Any]) -> None:
    """Feature interpretation pipeline summary."""
    if not data:
        out.write("exp22_feature_interpretation.json not found.\n")
        return

    _write_header(out, "Experiment 22 – Feature Interpretation Pipeline")

    feats = data.get("features", []) or []
    out.write(f"Features analyzed: {len(feats)}\n")

    hyp = data.get("hypothesis_test", {}) or {}
    if hyp:
        out.write(
            f"Spearman corr(monolinguality, monosemanticity)={hyp.get('spearman_correlation')}, "
            f"p={hyp.get('p_value')}, n={hyp.get('n_features')}\n"
        )
        if "random_baseline_mean" in hyp:
            out.write(
                f"Random baseline mean={hyp.get('random_baseline_mean')}, "
                f"high-mono mean={hyp.get('mean_monosemanticity')}, "
                f"improvement={hyp.get('improvement_over_random')}\n"
            )
            out.write(f"Baseline comparison: {hyp.get('baseline_comparison')}\n")
    out.write("\n")


def summarize_exp23(out, data: Dict[str, Any]) -> None:
    """Hierarchy causal validation summary."""
    if not data:
        out.write("exp23_hierarchy_causal.json not found.\n")
        return

    _write_header(out, "Experiment 23 – Hierarchy Causal Validation")

    v = data.get("validation", {}) or {}
    out.write(f"Hierarchy validated: {v.get('hierarchy_validated')}\n")
    out.write(f"Interpretation: {v.get('interpretation')}\n")
    out.write(f"Early script degradation: {v.get('early_script_degradation')}\n")
    out.write(f"Late semantic degradation: {v.get('late_semantic_degradation')}\n\n")


def summarize_exp24(out, data: Dict[str, Any]) -> None:
    """SAE detector summary."""
    if not data:
        out.write("exp24_sae_detector.json not found.\n")
        return

    _write_header(out, "Experiment 24 – SAE-Based Language Detector")

    out.write(f"Best layer: {data.get('best_layer')}, accuracy: {data.get('best_accuracy')}\n")
    rb = data.get("random_baseline", {}) or {}
    out.write(f"Random baseline: mean={rb.get('mean')}, std={rb.get('std')}\n")

    v = data.get("validation", {}) or {}
    out.write(f"Validated: {v.get('overall_validated')}\n\n")


def summarize_exp25(out, data: Dict[str, Any]) -> None:
    """Family causal ablation summary."""
    if not data:
        out.write("exp25_family_causal.json not found.\n")
        return

    _write_header(out, "Experiment 25 – Family Feature Causal Ablation")

    ca = data.get("causal_analysis", {}) or {}
    out.write(f"Family separation causal: {ca.get('family_separation_causal')}\n")
    out.write(f"Interpretation: {ca.get('interpretation')}\n")
    out.write(f"p_ia_holm={ca.get('p_ia_holm')}, p_dr_holm={ca.get('p_dr_holm')}\n")
    out.write(f"power_ia={ca.get('power_ia')}, power_dr={ca.get('power_dr')}\n\n")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    with open(REPORT_PATH, "w") as out:
        _write_header(out, "SAE Multilingual Steering – Summary Report")
        out.write("This report summarizes key metrics from all experiments.\n")
        out.write("All numbers are taken directly from JSON files in results/.\n")

        # Load JSONs (2B + optional 9B variants)
        exp1_v = _load_json_variants("exp1_feature_discovery")
        exp3_v = _load_json_variants("exp3_hindi_urdu_fixed")
        exp5_v = _load_json_variants("exp5_hierarchical_analysis")
        exp6_v = _load_json_variants("exp6_script_semantics_controls")
        exp4_v = _load_json_variants("exp4_spillover")
        exp8 = _load_json("exp8_scaling_9b_low_resource")  # already multi-variant
        exp9_v = _load_json_variants("exp9_layer_sweep_steering")
        exp10_v = _load_json_variants("exp10_attribution_occlusion")
        exp11_v = _load_json_variants("exp11_judge_calibration")
        exp12_v = _load_json_variants("exp12_qa_degradation")
        exp13_v = _load_json_variants("exp13_script_semantic_ablation")
        exp14_v = _load_json_variants("exp14_language_agnostic_space")
        exp15_v = _load_json_variants("exp15_steering_schedule")
        exp18_v = _load_json_variants("exp18_typological_features")
        exp19_v = _load_json_variants("exp19_crosslayer_causal")
        exp20_v = _load_json_variants("exp20_training_freq_control")
        exp21_v = _load_json_variants("exp21_family_separation")
        exp22_v = _load_json_variants("exp22_feature_interpretation")
        exp23_v = _load_json_variants("exp23_hierarchy_causal")
        exp24_v = _load_json_variants("exp24_sae_detector")
        exp25_v = _load_json_variants("exp25_family_causal")

        # ------------------------------------------------------------------
        # Paper assets: auto-generated LaTeX tables from JSON results.
        # ------------------------------------------------------------------
        for label, data in exp1_v.items():
            write_table_exp1_feature_discovery(data or {}, suffix="" if label == "2b" else "_9b")
        for label, data in exp3_v.items():
            write_table_exp3_hi_ur_overlap(data or {}, suffix="" if label == "2b" else "_9b")
        for label, data in exp5_v.items():
            write_table_exp5_hierarchy(data or {}, suffix="" if label == "2b" else "_9b")
        for label, data in exp6_v.items():
            write_table_exp6_script_semantics(data or {}, suffix="" if label == "2b" else "_9b")

        def _run_variant(label: str, fn, data: Dict[str, Any]):
            if label == "9b":
                out.write("\n[Model variant: Gemma-2-9B]\n")
            fn(out, data or {})

        for label, data in exp1_v.items():
            _run_variant(label, summarize_exp1, data)
        for label, data in exp3_v.items():
            _run_variant(label, summarize_exp3, data)
        for label, data in exp4_v.items():
            _run_variant(label, summarize_exp4, data)
        for label, data in exp5_v.items():
            _run_variant(label, summarize_exp5, data)
        for label, data in exp6_v.items():
            _run_variant(label, summarize_exp6, data)
        summarize_exp8(out, exp8 or {})
        for label, data in exp9_v.items():
            _run_variant(label, summarize_exp9, data)
        for label, data in exp10_v.items():
            _run_variant(label, summarize_exp10, data)
        for label, data in exp11_v.items():
            _run_variant(label, summarize_exp11, data)
        for label, data in exp12_v.items():
            _run_variant(label, summarize_exp12, data)
        for label, data in exp13_v.items():
            _run_variant(label, summarize_exp13, data)
        for label, data in exp14_v.items():
            _run_variant(label, summarize_exp14, data)
        for label, data in exp15_v.items():
            _run_variant(label, summarize_exp15, data)
        for label, data in exp18_v.items():
            _run_variant(label, summarize_exp18, data)
        for label, data in exp19_v.items():
            _run_variant(label, summarize_exp19, data)
        for label, data in exp20_v.items():
            _run_variant(label, summarize_exp20, data)
        for label, data in exp21_v.items():
            _run_variant(label, summarize_exp21, data)
        for label, data in exp22_v.items():
            _run_variant(label, summarize_exp22, data)
        for label, data in exp23_v.items():
            _run_variant(label, summarize_exp23, data)
        for label, data in exp24_v.items():
            _run_variant(label, summarize_exp24, data)
        for label, data in exp25_v.items():
            _run_variant(label, summarize_exp25, data)

    print(f"Summary report written to {REPORT_PATH}")
    print(f"LaTeX tables written to {TABLES_DIR}/ (when source JSONs exist)")


if __name__ == "__main__":
    main()
