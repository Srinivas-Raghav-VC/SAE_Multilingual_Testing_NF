"""Publication-quality plots for SAE multilingual steering experiments.

Generates all figures needed for paper submission.

Usage:
    python plots.py --results_dir results/
    
Outputs figures to: results/figures/
"""

import json
import argparse
from pathlib import Path
import numpy as np

# Check for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Install with: pip install seaborn")


# Publication-quality settings
FIGSIZE_SINGLE = (6, 4)
FIGSIZE_DOUBLE = (10, 4)
FIGSIZE_LARGE = (10, 8)
DPI = 300
FONT_SIZE = 12

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
        'legend.fontsize': FONT_SIZE - 2,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
    })
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")


def load_results(results_dir):
    """Load all experiment results from JSON files."""
    results_dir = Path(results_dir)
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            results[json_file.stem] = json.load(f)
    
    return results


def iter_variants(results, base_key):
    """Yield (label, suffix, data) for 2B/9B JSON variants if present."""
    base = results.get(base_key)
    if base:
        yield "2B", "", base
    v9 = results.get(f"{base_key}_9b")
    if v9:
        yield "9B", "_9b", v9


def _ordered_langs(lang_set):
    """Deterministic language ordering that highlights Indic vs controls."""
    preferred = ["hi", "ur", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "as", "en", "de", "fr", "es", "ar", "zh", "vi"]
    langs = [l for l in preferred if l in lang_set]
    for l in sorted(lang_set):
        if l not in langs:
            langs.append(l)
    return langs


def plot_exp1_selectivity_heatmap(results, output_dir):
    """Figure: Language-selective feature counts (Exp1).

    Heatmap of (# features with M > threshold) for each language×layer.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    for label, suffix, exp1 in iter_variants(results, "exp1_feature_discovery"):
        if not exp1:
            continue

        layers = sorted(int(k) for k in exp1.keys())
        if not layers:
            continue

        lang_set = set()
        for layer in layers:
            layer_data = exp1.get(str(layer), {})
            lang_set.update(layer_data.get("lang_features", {}).keys())
        langs = _ordered_langs(lang_set)
        if not langs:
            continue

        mat = np.zeros((len(langs), len(layers)), dtype=float)
        for i, lang in enumerate(langs):
            for j, layer in enumerate(layers):
                layer_data = exp1.get(str(layer), {})
                feats = layer_data.get("lang_features", {}).get(lang, [])
                mat[i, j] = float(len(feats))

        fig_w = max(6.5, 0.8 * len(layers) + 3.0)
        fig_h = max(4.5, 0.35 * len(langs) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            mat,
            xticklabels=layers,
            yticklabels=langs,
            cmap="YlGnBu",
            annot=(len(layers) <= 10 and len(langs) <= 12),
            fmt=".0f",
            cbar_kws={"label": "# language-selective features"},
            ax=ax,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Language")
        ax.set_title(f"Language-selective feature counts by layer (Exp1; {label})")

        output_path = output_dir / f"fig1_exp1_selectivity_heatmap{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


def plot_feature_counts_by_layer(results, output_dir):
    """Figure 1: Language-specific feature counts by layer.
    
    Bar chart showing how many features have M > 3.0 for each language at each layer.
    """
    if not HAS_MATPLOTLIB:
        return
    
    exp1 = results.get("exp1_feature_discovery", {})
    if not exp1:
        print("No exp1 results found")
        return
    
    # Keys are stringified layer numbers, e.g. "5", "8", ...
    layers = sorted(int(k) for k in exp1.keys())
    
    # Extract Hindi feature counts (M > threshold)
    hi_counts = []
    for layer in layers:
        layer_data = exp1.get(str(layer), {})
        lang_features = layer_data.get("lang_features", {})
        hi_feats = lang_features.get("hi", [])
        hi_count = len(hi_feats)
        hi_counts.append(hi_count)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    x = np.arange(len(layers))
    bars = ax.bar(x, hi_counts, color='#2ecc71', edgecolor='black', linewidth=1)
    
    # Highlight mid-range
    mid_range = [i for i, l in enumerate(layers) if 10 <= l <= 16]
    for i in mid_range:
        bars[i].set_color('#e74c3c')
        bars[i].set_label('Mid-layers (10-16)' if i == mid_range[0] else '')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Hindi-specific Features (M > 3.0)')
    ax.set_title('Language-Specific Feature Distribution Across Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, hi_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=10)
    
    ax.legend()
    
    output_path = output_dir / "fig1_feature_counts_by_layer.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hindi_urdu_overlap(results, output_dir):
    """Figure 3: Hindi-Urdu feature overlap across layers.
    
    Line plot showing Jaccard overlap between Hindi and Urdu features.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp3 in iter_variants(results, "exp3_hindi_urdu_fixed"):
        if not exp3:
            continue

        layers_data = exp3.get("layers", {})
        if not layers_data:
            print(f"exp3_hindi_urdu_fixed{suffix} has no 'layers' key")
            continue

        found = True
        layers = sorted(int(k) for k in layers_data.keys())

        hi_ur_overlap = []
        hi_en_overlap = []
        script_features_hi = []
        script_features_ur = []

        for layer in layers:
            layer_data = layers_data.get(str(layer), {})
            j = layer_data.get("jaccard_overlaps", {})
            s = layer_data.get("script_semantic", {})
            hi_ur_overlap.append(j.get("hindi_urdu", 0) * 100)
            hi_en_overlap.append(j.get("hindi_english", 0) * 100)
            script_features_hi.append(s.get("hindi_script", 0))
            script_features_ur.append(s.get("urdu_script", 0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        # Left: Overlap percentages
        ax1.plot(
            layers,
            hi_ur_overlap,
            "o-",
            color="#2ecc71",
            label="Hindi-Urdu",
            linewidth=2,
            markersize=8,
        )
        ax1.plot(
            layers,
            hi_en_overlap,
            "s-",
            color="#3498db",
            label="Hindi-English",
            linewidth=2,
            markersize=8,
        )
        ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50%")

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Jaccard overlap (%)")
        ax1.set_title(f"Feature overlap across layers (Exp3; {label})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 110)

        # Right: Script-only feature counts
        x = np.arange(len(layers))
        width = 0.35

        ax2.bar(
            x - width / 2,
            script_features_hi,
            width,
            label="Hindi (Devanagari)",
            color="#e74c3c",
        )
        ax2.bar(
            x + width / 2,
            script_features_ur,
            width,
            label="Urdu (Arabic)",
            color="#9b59b6",
        )

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Script-only feature count")
        ax2.set_title("Script-only features by layer")
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers)
        ax2.legend()

        plt.tight_layout()
        output_path = output_dir / f"fig3_hindi_urdu_overlap{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp3 results found")


def plot_script_semantics_controls(results, output_dir):
    """Figure 4: Script vs semantic feature counts by layer (Exp6).
    
    Uses exp6_script_semantics_controls.json to show how many features
    are script-only vs semantic vs family-semantic across layers.
    """
    if not HAS_MATPLOTLIB:
        return
    
    found = False
    for label, suffix, exp6 in iter_variants(results, "exp6_script_semantics_controls"):
        if not exp6:
            continue
        found = True

        layers = sorted(int(k) for k in exp6.keys())
        hi_script = [exp6[str(l)]["hi_script_only"] for l in layers]
        hi_sem = [exp6[str(l)]["hi_semantic"] for l in layers]
        fam_sem = [exp6[str(l)]["family_semantic"] for l in layers]

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        x = np.arange(len(layers))
        width = 0.25

        ax.bar(x - width, hi_script, width, label="HI script-only", color="#e74c3c")
        ax.bar(x, hi_sem, width, label="HI semantic", color="#2ecc71")
        ax.bar(x + width, fam_sem, width, label="Family semantic", color="#3498db")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Feature count")
        ax.set_title(f"Script vs semantic feature counts (Exp6; {label})")
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()

        output_path = output_dir / f"fig4_script_semantics_controls{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

        # Optional: if Exp6 stored Jaccard controls, plot them as a second panel.
        j_key = "jaccard_hi_deva_hi_latin"
        if all(j_key in exp6.get(str(l), {}) for l in layers):
            j_hi_latin = [float(exp6[str(l)].get("jaccard_hi_deva_hi_latin", 0.0)) * 100.0 for l in layers]
            j_hi_ur = [float(exp6[str(l)].get("jaccard_hi_deva_ur", 0.0)) * 100.0 for l in layers]
            j_hi_bn = [float(exp6[str(l)].get("jaccard_hi_deva_bn", 0.0)) * 100.0 for l in layers]
            j_hi_noise = [float(exp6[str(l)].get("jaccard_hi_deva_noise", 0.0)) * 100.0 for l in layers]

            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.plot(layers, j_hi_latin, "o-", label="J(HI_Deva, HI_Latin)")
            ax.plot(layers, j_hi_ur, "s-", label="J(HI_Deva, UR)")
            ax.plot(layers, j_hi_bn, "^-", label="J(HI_Deva, BN)")
            ax.plot(layers, j_hi_noise, "x--", label="J(HI_Deva, Noise)", alpha=0.8)

            ax.set_xlabel("Layer")
            ax.set_ylabel("Jaccard overlap (%)")
            ax.set_title(f"Transliteration/noise controls (Exp6; {label})")
            ax.grid(True, alpha=0.3)
            ax.legend()

            output_path = output_dir / f"fig4b_script_semantics_jaccard{suffix}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")

    if not found:
        print("No exp6 results found")


def plot_monolinguality_distribution(results, output_dir):
    """Figure 5: Distribution of monolinguality scores.
    
    Histogram showing how monolinguality scores are distributed.
    """
    if not HAS_MATPLOTLIB:
        return
    
    print(
        "Skipping monolinguality distribution: exp1_feature_discovery.json "
        "does not include per-feature monolinguality scores."
    )
    return


def plot_dead_features(results, output_dir):
    """Figure 6: Dead feature analysis.
    
    Bar chart showing dead features by language.
    """
    if not HAS_MATPLOTLIB:
        return
    
    exp1 = results.get("exp1_feature_discovery", {})
    
    # Look for dead_features in any layer data
    dead_features = None
    for key, value in exp1.items():
        if isinstance(value, dict) and "dead_features" in value:
            dead_features = value["dead_features"]
            break
    
    if not dead_features:
        print("No dead feature data found")
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    languages = list(dead_features.keys())
    counts = list(dead_features.values())
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    bars = ax.bar(languages, counts, color=colors[:len(languages)], edgecolor='black')
    
    ax.set_xlabel('Language')
    ax.set_ylabel('Dead Feature Count')
    ax.set_title(f'Dead Features by Language (Total SAE features: 16384)')
    
    # Add percentage labels
    total = 16384
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    output_path = output_dir / "fig5_dead_features.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_table(results, output_dir):
    """Table 1: Summary of hypothesis testing results.
    
    Creates a summary figure with hypothesis results.
    """
    print(
        "Skipping hypothesis summary table: the previous version hard-coded "
        "numbers and can easily become stale/misleading."
    )
    return


def plot_feature_labels_summary(results, output_dir):
    """Figure 7: Summary of Gemini feature labels (Exp5).
    
    Creates a simple table-like figure showing a few labeled features
    across layers to give intuition about learned concepts.
    """
    if not HAS_MATPLOTLIB:
        return
    
    exp5 = results.get("exp5_hierarchical_analysis", {})
    if not exp5:
        print("No exp5_hierarchical_analysis results found")
        return
    
    labels = exp5.get("feature_labels", [])
    if not labels:
        print("No feature_labels found in exp5_hierarchical_analysis")
        return
    
    # Take at most 10 labels for readability
    rows = labels[:10]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    
    columns = ["Layer", "Feature", "Label", "Languages", "Conf."]
    data = [
        [
            r.get("layer", ""),
            r.get("feature_idx", ""),
            r.get("label", "")[:30],
            ",".join(r.get("languages", []))[:20],
            f"{r.get('confidence', 0.0):.2f}",
        ]
        for r in rows
    ]
    
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    
    ax.set_title("Sample Gemini-Labeled Features (Exp5)", fontsize=12, pad=10)
    
    output_path = output_dir / "fig7_feature_labels_summary.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_language_overlap_heatmaps(results, output_dir):
    """Figure 8: Language overlap heatmaps (Exp5, early/mid/late).
    
    Uses exp5_hierarchical_analysis.json to build language×language
    Jaccard overlap matrices averaged over early, mid, and late layers.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    found = False
    for label, suffix, exp5 in iter_variants(results, "exp5_hierarchical_analysis"):
        if not exp5:
            continue

        analyses = exp5.get("layer_analyses", [])
        if not analyses:
            print(f"No layer_analyses in exp5_hierarchical_analysis{suffix}")
            continue

        layers = sorted({int(e.get("layer")) for e in analyses if e.get("layer") is not None})
        if not layers:
            continue
        max_layer = max(layers)

        # Collect language codes from overlap keys like "hi-ur"
        lang_set = set()
        for entry in analyses:
            overlaps = entry.get("overlaps", {})
            for key in overlaps.keys():
                parts = key.split("-")
                if len(parts) == 2:
                    lang_set.update(parts)
        if not lang_set:
            continue

        found = True
        langs = _ordered_langs(lang_set)
        lang_idx = {l: i for i, l in enumerate(langs)}

        def layer_group(layer: int) -> str:
            frac = float(layer) / float(max_layer) if max_layer > 0 else 0.0
            if frac <= 0.33:
                return "early"
            if frac <= 0.66:
                return "middle"
            return "late"

        groups = {
            "early": {"layers": [], "sum": np.zeros((len(langs), len(langs)), dtype=float), "n": 0},
            "middle": {"layers": [], "sum": np.zeros((len(langs), len(langs)), dtype=float), "n": 0},
            "late": {"layers": [], "sum": np.zeros((len(langs), len(langs)), dtype=float), "n": 0},
        }

        for entry in analyses:
            layer = entry.get("layer", None)
            if layer is None:
                continue
            group = layer_group(int(layer))
            overlaps = entry.get("overlaps", {})

            mat = np.zeros((len(langs), len(langs)), dtype=float)
            for key, val in overlaps.items():
                parts = key.split("-")
                if len(parts) != 2:
                    continue
                a, b = parts
                if a not in lang_idx or b not in lang_idx:
                    continue
                mat[lang_idx[a], lang_idx[b]] = float(val)

            # Symmetrise and set diagonal to 1.0 for readability.
            mat = (mat + mat.T) / 2.0
            np.fill_diagonal(mat, 1.0)

            groups[group]["layers"].append(int(layer))
            groups[group]["sum"] += mat
            groups[group]["n"] += 1

        for group_name, g in groups.items():
            if g["n"] <= 0:
                continue
            avg = g["sum"] / float(g["n"])

            # Clustered heatmap makes "Indic cluster vs controls" easy to see.
            cg = sns.clustermap(
                avg,
                xticklabels=langs,
                yticklabels=langs,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                figsize=(9, 9),
                cbar_kws={"label": "Jaccard overlap"},
            )
            cg.fig.suptitle(
                f"Language overlap clustering ({group_name}; Exp5; {label})",
                y=1.02,
            )

            output_path = output_dir / f"fig8_language_overlap_clustermap_{group_name}{suffix}.png"
            cg.fig.savefig(output_path)
            plt.close(cg.fig)
            print(f"Saved: {output_path}")

    if not found:
        print("No exp5_hierarchical_analysis results found")


def plot_steering_profile_similarity(results, output_dir):
    """Figure 9: Steering profile similarity across languages (Exp9).
    
    Uses exp9_layer_sweep_steering.json. For each language, builds a
    vector of best script+semantic success per layer (activation_diff
    method), then computes Pearson correlation between languages and
    plots a heatmap.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    found = False
    for label, suffix, exp9 in iter_variants(results, "exp9_layer_sweep_steering"):
        if not exp9:
            continue

        langs = sorted(exp9.keys())
        if not langs:
            continue

        # Determine common set of layers
        all_layers = set()
        for lang in langs:
            layers_d = exp9[lang].get("layers", {})
            all_layers.update(int(l) for l in layers_d.keys())
        layers = sorted(all_layers)
        if not layers:
            continue

        # Build per-language steering profile: best success per layer
        profiles = []
        valid_langs = []

        for lang in langs:
            lang_layers = exp9[lang].get("layers", {})
            if not lang_layers:
                continue
            vec = []
            for layer in layers:
                layer_str = str(layer)
                if layer_str not in lang_layers:
                    vec.append(0.0)
                    continue
                methods = lang_layers[layer_str]
                if "activation_diff" not in methods:
                    vec.append(0.0)
                    continue
                meth = methods["activation_diff"]
                best = 0.0
                for _strength_str, metrics in meth.items():
                    # Prefer script+semantic if available, else script-only
                    ss = metrics.get("success_rate_script_semantic", None)
                    if ss is None:
                        ss = metrics.get("success_rate_script", 0.0)
                    if ss is None:
                        ss = 0.0
                    best = max(best, float(ss))
                vec.append(best)
            profiles.append(vec)
            valid_langs.append(lang)

        if len(profiles) < 2:
            continue

        profiles_arr = np.array(profiles)  # shape: (n_langs, n_layers)
        if np.all(profiles_arr == 0):
            continue

        found = True
        corr = np.corrcoef(profiles_arr)

        fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
        sns.heatmap(
            corr,
            xticklabels=valid_langs,
            yticklabels=valid_langs,
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            annot=False,
            ax=ax,
        )
        ax.set_title(f"Steering profile similarity (activation_diff; Exp9; {label})")
        ax.set_xlabel("Language")
        ax.set_ylabel("Language")

        output_path = output_dir / f"fig9_steering_profile_similarity{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp9_layer_sweep_steering results found")


def plot_exp9_layer_method_heatmaps(results, output_dir):
    """Figure 9b: Layer×method steering heatmaps per language (Exp9).

    For each language in exp9_layer_sweep_steering.json, compute the best
    success rate across strengths for each (layer, method) and plot a heatmap.

    This is the main empirical visualization of the "generator window":
    reviewers can directly see which layers/methods work and verify that
    very-late layers (e.g., 24) are ineffective.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    found = False
    for label, suffix, exp9 in iter_variants(results, "exp9_layer_sweep_steering"):
        if not exp9:
            continue
        found = True

        for lang, lang_data in exp9.items():
            layers_d = lang_data.get("layers", {})
            if not layers_d:
                continue

            layers = sorted(int(k) for k in layers_d.keys())
            methods = sorted({m for d in layers_d.values() for m in d.keys()})
            if not layers or not methods:
                continue

            mat = np.zeros((len(layers), len(methods)), dtype=float)

            for i, layer in enumerate(layers):
                per_method = layers_d.get(str(layer), {})
                for j, method in enumerate(methods):
                    per_strength = per_method.get(method, {})
                    best = 0.0
                    for _strength, metrics in per_strength.items():
                        ss = metrics.get("success_rate_script_semantic", None)
                        if ss is None:
                            ss = metrics.get("success_rate_script", 0.0)
                        if ss is None:
                            ss = 0.0
                        best = max(best, float(ss))
                    mat[i, j] = best

            fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
            sns.heatmap(
                mat,
                xticklabels=methods,
                yticklabels=layers,
                vmin=0.0,
                vmax=1.0,
                cmap="YlGnBu",
                annot=False,
                ax=ax,
            )
            ax.set_xlabel("Steering method")
            ax.set_ylabel("Layer")
            ax.set_title(f"Steering success heatmap (Exp9; {label}): {lang.upper()}")

            output_path = output_dir / f"fig9_steering_heatmap_{lang}{suffix}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")

    if not found:
        print("No exp9_layer_sweep_steering results found")


def plot_hierarchy_profile_similarity(results, output_dir):
    """Figure 10: Hierarchical representation profile similarity (Exp5).
    
    Uses exp5_hierarchical_analysis.json. For each language, builds a
    vector of active-feature counts per layer (from per_language) and
    computes Pearson correlation between languages. This shows whether
    languages share a similar *hierarchical* layer profile.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    found = False
    for label, suffix, exp5 in iter_variants(results, "exp5_hierarchical_analysis"):
        if not exp5:
            continue

        analyses = exp5.get("layer_analyses", [])
        if not analyses:
            continue

        # Collect languages and layers
        lang_set = set()
        layers = []
        for entry in analyses:
            layer = entry.get("layer", None)
            if layer is None:
                continue
            layers.append(layer)
            per_lang = entry.get("per_language", {})
            lang_set.update(per_lang.keys())

        if not lang_set or not layers:
            continue

        found = True
        layers = sorted(set(layers))
        langs = sorted(lang_set)

        # Build per-language hierarchy profile: active-feature counts per layer
        profiles = []
        valid_langs = []

        for lang in langs:
            vec = []
            for layer in layers:
                # Find the entry for this layer
                count = 0.0
                for entry in analyses:
                    if entry.get("layer", None) == layer:
                        per_lang = entry.get("per_language", {})
                        count = float(per_lang.get(lang, 0))
                        break
                vec.append(count)

            if all(v == 0.0 for v in vec):
                continue
            profiles.append(vec)
            valid_langs.append(lang)

        if len(profiles) < 2:
            continue

        profiles_arr = np.array(profiles)  # (n_langs, n_layers)
        # Normalize each language profile (center and scale) to focus on shape
        profiles_norm = (profiles_arr - profiles_arr.mean(axis=1, keepdims=True))
        std = profiles_norm.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        profiles_norm = profiles_norm / std

        corr = np.corrcoef(profiles_norm)

        fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
        sns.heatmap(
            corr,
            xticklabels=valid_langs,
            yticklabels=valid_langs,
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            annot=False,
            ax=ax,
        )
        ax.set_title(f"Hierarchy profile similarity (Exp5; {label})")
        ax.set_xlabel("Language")
        ax.set_ylabel("Language")

        output_path = output_dir / f"fig10_hierarchy_profile_similarity{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp5_hierarchical_analysis results found")


def plot_language_agnostic_alignment(results, output_dir):
    """Figure 11: EN–Indic alignment vs layer (Exp14).

    Uses exp14_language_agnostic_space.json to plot average cosine similarity
    between English and each Indic language across layers.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp14 in iter_variants(results, "exp14_language_agnostic_space"):
        if not exp14:
            continue

        layers_data = exp14.get("layers", {})
        if not layers_data:
            continue

        found = True
        layers = sorted(int(k) for k in layers_data.keys())
        en_hi = []
        en_ur = []
        en_bn = []
        en_ta = []
        en_te = []

        for layer in layers:
            pair = layers_data[str(layer)].get("pairwise_cosine", {})
            en_hi.append(pair.get("en-hi", 0.0))
            en_ur.append(pair.get("en-ur", 0.0))
            en_bn.append(pair.get("en-bn", 0.0))
            en_ta.append(pair.get("en-ta", 0.0))
            en_te.append(pair.get("en-te", 0.0))

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(layers, en_hi, "o-", label="EN–HI")
        ax.plot(layers, en_ur, "s-", label="EN–UR")
        ax.plot(layers, en_bn, "^-", label="EN–BN")
        ax.plot(layers, en_ta, "D-", label="EN–TA")
        ax.plot(layers, en_te, "x-", label="EN–TE")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Average cosine similarity")
        ax.set_title(f"Cross-lingual alignment vs layer (Exp14; {label})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        output_path = output_dir / f"fig11_language_agnostic_alignment{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp14_language_agnostic_space results found")


def plot_spillover_analysis(results, output_dir):
    """Figure for Exp4: Spillover analysis (EN→HI vs EN→DE control)."""

    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp4 in iter_variants(results, "exp4_spillover"):
        if not exp4:
            continue
        found = True

        def get_dist(condition_key):
            layers = exp4.get(condition_key, {})
            if not layers:
                return None

            # Prefer layer 20 (legacy 2B runs), else pick the max available layer.
            layer = "20" if "20" in layers else max(layers.keys(), key=int)
            strengths = layers.get(layer, {})
            if not strengths:
                return None

            # Prefer strength 2.0, else max strength.
            strength = "2.0" if "2.0" in strengths else max(strengths.keys(), key=float)
            return strengths.get(strength, {}).get("language_distribution", {})

        dist_hi = get_dist("en_to_hi_layers")
        dist_de = get_dist("en_to_de_layers")

        if not dist_hi or not dist_de:
            print(f"Insufficient data for Exp4 plot{suffix}")
            continue

        langs = ["hi", "ur", "bn", "en", "de"]
        labels_x = ["Hindi", "Urdu", "Bengali", "English", "German"]

        vals_hi = [dist_hi.get(l, 0.0) for l in langs]
        vals_de = [dist_de.get(l, 0.0) for l in langs]

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        x = np.arange(len(langs))
        width = 0.35

        ax.bar(x - width / 2, vals_hi, width, label="Steer EN→HI", color="#e74c3c")
        ax.bar(x + width / 2, vals_de, width, label="Steer EN→DE (control)", color="#95a5a6")

        ax.set_ylabel("Output %")
        ax.set_title(f"Spillover: target vs control steering (Exp4; {label})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        output_path = output_dir / f"fig_exp4_spillover{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp4_spillover results found")


def plot_exp11_judge_calibration(results, output_dir):
    """Appendix figure: raw vs calibrated judge accuracies (Exp11)."""
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp11 in iter_variants(results, "exp11_judge_calibration"):
        if not exp11:
            continue
        langs_block = exp11.get("languages", {})
        if not langs_block:
            continue
        found = True

        langs = sorted(langs_block.keys())
        raw = [float(langs_block[l].get("raw_accuracy", 0.0)) * 100.0 for l in langs]
        corr = [float(langs_block[l].get("corrected_accuracy", 0.0)) * 100.0 for l in langs]

        ci_low = []
        ci_high = []
        for l in langs:
            ci = langs_block[l].get("confidence_interval", None)
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                ci_low.append(float(ci[0]) * 100.0)
                ci_high.append(float(ci[1]) * 100.0)
            else:
                ci_low.append(np.nan)
                ci_high.append(np.nan)

        x = np.arange(len(langs))
        width = 0.38

        fig_w = max(7.0, 0.7 * len(langs) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))
        ax.bar(x - width / 2, raw, width, label="Raw", color="#95a5a6")
        ax.bar(x + width / 2, corr, width, label="Calibrated", color="#2ecc71")

        # Error bars for calibrated estimates (if present)
        yerr_lo = np.array(corr) - np.array(ci_low)
        yerr_hi = np.array(ci_high) - np.array(corr)
        if np.isfinite(yerr_lo).any() and np.isfinite(yerr_hi).any():
            ax.errorbar(
                x + width / 2,
                corr,
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in langs])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Judge calibration (Exp11; {label})")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        output_path = output_dir / f"figA1_exp11_judge_calibration{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp11_judge_calibration results found")


def plot_exp12_qa_deltas(results, output_dir):
    """Figure: QA impact under steering (Exp12)."""
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp12 in iter_variants(results, "exp12_qa_degradation"):
        if not exp12:
            continue

        rows = []
        for key, kd in exp12.items():
            if not isinstance(kd, dict):
                continue
            base = kd.get("baseline", {})
            steer = kd.get("steered", {})
            if not base or not steer:
                continue

            if "_" in key:
                task, lang = key.split("_", 1)
            else:
                task, lang = key, "??"

            qa_f1_base = base.get("qa_f1", None)
            qa_f1_steer = steer.get("qa_f1", None)
            if qa_f1_base is None or qa_f1_steer is None:
                continue

            delta_f1 = float(qa_f1_steer) - float(qa_f1_base)
            base_pres = steer.get("baseline_preservation_semantic_mean", None)
            rows.append(
                {
                    "label": f"{task}:{lang}",
                    "delta_f1": delta_f1,
                    "base_pres": float(base_pres) if isinstance(base_pres, (int, float)) else np.nan,
                }
            )

        if not rows:
            continue

        found = True
        rows = sorted(rows, key=lambda r: r["delta_f1"])
        labels_y = [r["label"] for r in rows]
        vals = [r["delta_f1"] * 100.0 for r in rows]

        fig_h = max(3.5, 0.35 * len(rows) + 1.5)
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        y = np.arange(len(rows))
        ax.barh(y, vals, color=["#e74c3c" if v < 0 else "#2ecc71" for v in vals])
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_y)
        ax.set_xlabel("Δ QA F1 (steered − baseline), percentage points")
        ax.set_title(f"QA impact under steering (Exp12; {label})")
        ax.grid(axis="x", alpha=0.3)

        output_path = output_dir / f"fig_exp12_qa_delta_f1{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp12_qa_degradation results found")





def plot_exp18_typological_features(results, output_dir):
    """Figure: Typological feature clustering analysis (Exp18).

    Shows separation ratios for retroflex, SOV, and family-based groupings.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp18 in iter_variants(results, "exp18_typological_features"):
        if not exp18:
            continue
        found = True

        typo = exp18.get("typological_analyses", {})
        family = exp18.get("family_analyses", {}).get("indic_families", {})

        if not typo and not family:
            continue

        # Build comparison data
        categories = []
        separations = []
        within_overlaps = []
        between_overlaps = []

        if "retroflex" in typo:
            categories.append("Retroflex")
            separations.append(typo["retroflex"].get("separation", 0))
            within_overlaps.append(typo["retroflex"].get("within_overlap", 0))
            between_overlaps.append(typo["retroflex"].get("between_overlap", 0))

        if "sov_order" in typo:
            categories.append("SOV Order")
            separations.append(typo["sov_order"].get("separation", 0))
            within_overlaps.append(typo["sov_order"].get("within_overlap", 0))
            between_overlaps.append(typo["sov_order"].get("between_overlap", 0))

        if family:
            categories.append("Family (IA vs DR)")
            ia_overlap = family.get("indo_aryan_overlap", 0)
            dr_overlap = family.get("dravidian_overlap", 0)
            cross = family.get("cross_family_overlap", 0.01)
            within_overlaps.append((ia_overlap + dr_overlap) / 2)
            between_overlaps.append(cross)
            separations.append(family.get("separation_ratio", 0))

        if not categories:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        # Left: Separation ratios
        x = np.arange(len(categories))
        ax1.bar(x, separations, color="#2ecc71", edgecolor="black")
        ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="No separation")
        ax1.axhline(y=1.5, color="red", linestyle="--", alpha=0.7, label="Clear separation")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=15)
        ax1.set_ylabel("Separation ratio (within/between)")
        ax1.set_title(f"Typological clustering (Exp18; {label})")
        ax1.legend(fontsize=8)

        # Right: Within vs between overlaps
        width = 0.35
        ax2.bar(x - width/2, within_overlaps, width, label="Within-group", color="#3498db")
        ax2.bar(x + width/2, between_overlaps, width, label="Between-group", color="#e74c3c")
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=15)
        ax2.set_ylabel("Jaccard overlap")
        ax2.set_title("Within vs between group overlap")
        ax2.legend()

        plt.tight_layout()
        output_path = output_dir / f"fig_exp18_typological{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp18_typological_features results found")


def plot_exp19_causal_profile(results, output_dir):
    """Figure: Cross-layer causal profiles (Exp19).

    Shows causal importance (ablation impact) across layers for different languages.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp19 in iter_variants(results, "exp19_crosslayer_causal"):
        if not exp19:
            continue

        profiles = exp19.get("causal_profiles", {})
        if not profiles:
            continue

        found = True

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        colors = {"hi": "#e74c3c", "ta": "#3498db", "te": "#2ecc71", "bn": "#9b59b6"}

        for lang, lang_profiles in profiles.items():
            layers = [p["layer"] for p in lang_profiles]
            importances = [p["causal_importance"] for p in lang_profiles]
            color = colors.get(lang, "#95a5a6")
            ax.plot(layers, importances, "o-", label=lang.upper(), color=color, linewidth=2)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Causal importance (|baseline - ablated|)")
        ax.set_title(f"Cross-layer causal profiles (Exp19; {label})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = output_dir / f"fig_exp19_causal_profile{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp19_crosslayer_causal results found")


def plot_exp20_frequency_control(results, output_dir):
    """Figure: Training frequency vs feature count analysis (Exp20).

    Scatter plot of perplexity (inverse frequency) vs feature count.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp20 in iter_variants(results, "exp20_training_freq_control"):
        if not exp20:
            continue

        freq = exp20.get("frequency_estimates", {})
        feats = exp20.get("feature_counts", {})
        corr = exp20.get("correlation_analysis", {})

        if not freq or not feats:
            continue

        found = True

        # Build scatter data
        langs = [l for l in freq if l in feats]
        ppls = [freq[l] for l in langs]
        counts = [feats[l] for l in langs]

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        # Color by family
        from config import INDIC_LANGUAGES
        ia = set(INDIC_LANGUAGES.get("indo_aryan", []))
        dr = set(INDIC_LANGUAGES.get("dravidian", []))

        for i, lang in enumerate(langs):
            if lang in ia:
                color = "#e74c3c"
            elif lang in dr:
                color = "#3498db"
            else:
                color = "#95a5a6"
            ax.scatter(ppls[i], counts[i], c=color, s=100, edgecolor="black")
            ax.annotate(lang.upper(), (ppls[i], counts[i]), fontsize=8,
                       xytext=(5, 5), textcoords="offset points")

        # Add legend
        ax.scatter([], [], c="#e74c3c", label="Indo-Aryan", s=100, edgecolor="black")
        ax.scatter([], [], c="#3498db", label="Dravidian", s=100, edgecolor="black")
        ax.scatter([], [], c="#95a5a6", label="Control", s=100, edgecolor="black")

        r2 = corr.get("r_squared", 0)
        ax.set_xlabel("Perplexity (higher = less training data)")
        ax.set_ylabel("Active feature count")
        ax.set_title(f"Training frequency vs features (R²={r2:.2f}; Exp20; {label})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = output_dir / f"fig_exp20_frequency_control{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp20_training_freq_control results found")


def plot_exp21_family_separation(results, output_dir):
    """Figure: Indo-Aryan vs Dravidian family separation (Exp21).

    Shows within-family vs cross-family overlaps and cross-family steering transfer.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp21 in iter_variants(results, "exp21_family_separation"):
        if not exp21:
            continue

        overlaps = exp21.get("family_overlaps", {})
        steering = exp21.get("cross_family_steering", [])

        if not overlaps:
            continue

        found = True

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        # Left: Family overlaps
        ax1 = axes[0]
        categories = ["Indo-Aryan\ninternal", "Dravidian\ninternal", "Cross-family"]
        values = [
            overlaps.get("indo_aryan_internal", 0),
            overlaps.get("dravidian_internal", 0),
            overlaps.get("cross_family", 0),
        ]
        colors = ["#e74c3c", "#3498db", "#95a5a6"]

        ax1.bar(categories, values, color=colors, edgecolor="black")
        ax1.set_ylabel("Jaccard overlap")
        ax1.set_title(f"Family overlap analysis (Exp21; {label})")

        sep_ratio = overlaps.get("separation_ratio", 0)
        ax1.text(0.5, 0.95, f"Separation ratio: {sep_ratio:.2f}",
                transform=ax1.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat"))

        # Right: Cross-family steering transfer
        ax2 = axes[1]
        if steering:
            pairs = [f"{s['source']}->{s['target']}" for s in steering]
            success = [s["success_rate"] * 100 for s in steering]

            y = np.arange(len(pairs))
            ax2.barh(y, success, color="#2ecc71", edgecolor="black")
            ax2.set_yticks(y)
            ax2.set_yticklabels(pairs)
            ax2.set_xlabel("Success rate (%)")
            ax2.set_title("Cross-family steering transfer")
            ax2.axvline(x=30, color="red", linestyle="--", alpha=0.7,
                       label="Transfer threshold (30%)")
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No steering data", ha="center", va="center",
                    transform=ax2.transAxes)
            ax2.set_title("Cross-family steering transfer")

        plt.tight_layout()
        output_path = output_dir / f"fig_exp21_family_separation{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp21_family_separation results found")


def plot_family_transfer_matrix(results, output_dir):
    """Figure: N×N language transfer matrix (Exp21 enhanced).

    Shows steering transfer effectiveness between all language pairs,
    organized by family (Indo-Aryan, Dravidian, Control).
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    found = False
    for label, suffix, exp21 in iter_variants(results, "exp21_family_separation"):
        if not exp21:
            continue

        steering = exp21.get("cross_family_steering", [])
        if not steering:
            continue

        found = True

        # Build transfer matrix
        langs = sorted(set(
            [s["source"] for s in steering] + [s["target"] for s in steering]
        ))
        lang_idx = {l: i for i, l in enumerate(langs)}
        n = len(langs)

        mat = np.zeros((n, n))
        np.fill_diagonal(mat, 1.0)  # Self-transfer is perfect

        for s in steering:
            src = s["source"]
            tgt = s["target"]
            if src in lang_idx and tgt in lang_idx:
                mat[lang_idx[src], lang_idx[tgt]] = s.get("success_rate", 0)

        # Reorder by family
        ia = ["hi", "ur", "bn", "mr", "gu", "pa"]
        dr = ["ta", "te", "kn", "ml"]
        ctrl = ["en", "de", "fr", "es"]

        ordered_langs = []
        for l in ia:
            if l in langs:
                ordered_langs.append(l)
        for l in dr:
            if l in langs:
                ordered_langs.append(l)
        for l in ctrl:
            if l in langs:
                ordered_langs.append(l)
        for l in langs:
            if l not in ordered_langs:
                ordered_langs.append(l)

        # Reorder matrix
        new_idx = [lang_idx[l] for l in ordered_langs]
        mat_ordered = mat[np.ix_(new_idx, new_idx)]

        fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
        sns.heatmap(
            mat_ordered * 100,
            xticklabels=[l.upper() for l in ordered_langs],
            yticklabels=[l.upper() for l in ordered_langs],
            vmin=0,
            vmax=100,
            cmap="YlGnBu",
            annot=len(ordered_langs) <= 8,
            fmt=".0f",
            cbar_kws={"label": "Transfer success (%)"},
            ax=ax,
        )
        ax.set_xlabel("Target language")
        ax.set_ylabel("Source language")
        ax.set_title(f"Cross-lingual steering transfer matrix (Exp21; {label})")

        # Add family boundary lines
        ia_count = sum(1 for l in ordered_langs if l in ia)
        dr_count = sum(1 for l in ordered_langs if l in dr)

        if ia_count > 0 and dr_count > 0:
            ax.axhline(y=ia_count, color="red", linewidth=2, linestyle="--")
            ax.axvline(x=ia_count, color="red", linewidth=2, linestyle="--")
            ax.axhline(y=ia_count + dr_count, color="blue", linewidth=2, linestyle="--")
            ax.axvline(x=ia_count + dr_count, color="blue", linewidth=2, linestyle="--")

        output_path = output_dir / f"fig_exp21_transfer_matrix{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp21_family_separation results found for transfer matrix")


def plot_monosemanticity_distribution(results, output_dir):
    """Figure: Monosemanticity score distribution by layer (Exp22).

    Shows histogram of coherence scores for top features at each layer.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp22 in iter_variants(results, "exp22_feature_interpretation"):
        if not exp22:
            continue

        features_db = exp22.get("features_db", [])
        if not features_db:
            continue

        found = True

        # Group by layer
        by_layer = {}
        for f in features_db:
            layer = f.get("layer")
            coherence = f.get("coherence", 0)
            if layer is not None:
                by_layer.setdefault(layer, []).append(coherence)

        if not by_layer:
            continue

        layers = sorted(by_layer.keys())

        # Create subplots
        n_cols = min(3, len(layers))
        n_rows = (len(layers) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.array(axes).flatten() if len(layers) > 1 else [axes]

        for i, layer in enumerate(layers):
            ax = axes[i]
            scores = by_layer[layer]

            ax.hist(scores, bins=20, color="#3498db", edgecolor="black", alpha=0.7)
            ax.axvline(x=np.mean(scores), color="red", linestyle="--",
                      label=f"Mean: {np.mean(scores):.2f}")
            ax.set_xlabel("Coherence score")
            ax.set_ylabel("Feature count")
            ax.set_title(f"Layer {layer}")
            ax.legend(fontsize=8)

        # Hide unused axes
        for j in range(len(layers), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Monosemanticity distribution (Exp22; {label})", fontsize=14)
        plt.tight_layout()

        output_path = output_dir / f"fig_exp22_monosemanticity{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp22_feature_interpretation results found")


def plot_hierarchy_causal_profile(results, output_dir):
    """Figure: Hierarchy causal validation (Exp23).

    Shows script vs semantic degradation when ablating features at
    early, mid, and late layers.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp23 in iter_variants(results, "exp23_hierarchy_causal"):
        if not exp23:
            continue

        validation = exp23.get("validation", {})
        if not validation:
            continue

        found = True

        # Extract degradation values
        groups = ["Early", "Mid", "Late"]
        script_deg = [
            validation.get("early_script_degradation", 0) * 100,
            validation.get("mid_script_degradation", 0) * 100,
            validation.get("late_script_degradation", 0) * 100,
        ]
        semantic_deg = [
            validation.get("early_semantic_degradation", 0) * 100,
            validation.get("mid_semantic_degradation", 0) * 100,
            validation.get("late_semantic_degradation", 0) * 100,
        ]

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        x = np.arange(len(groups))
        width = 0.35

        bars1 = ax.bar(x - width/2, script_deg, width, label="Script degradation",
                       color="#e74c3c", edgecolor="black")
        bars2 = ax.bar(x + width/2, semantic_deg, width, label="Semantic degradation",
                       color="#3498db", edgecolor="black")

        ax.set_xlabel("Layer group")
        ax.set_ylabel("Degradation (%)")
        ax.set_title(f"Hierarchy causal validation (Exp23; {label})")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add hypothesis annotation
        validated = validation.get("hierarchy_validated", False)
        interp = validation.get("interpretation", "")
        status = "VALIDATED" if validated else "NOT VALIDATED"
        color = "#2ecc71" if validated else "#e74c3c"

        ax.text(0.5, 0.95, f"Hierarchy: {status}",
               transform=ax.transAxes, ha="center", fontsize=11,
               bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))

        output_path = output_dir / f"fig_exp23_hierarchy_causal{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp23_hierarchy_causal results found")


def plot_sae_detector_analysis(results, output_dir):
    """Figure: SAE-based language detector analysis (Exp24).

    Shows classifier accuracy by layer and steering prediction shift.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp24 in iter_variants(results, "exp24_sae_detector"):
        if not exp24:
            continue

        clf_results = exp24.get("classifier_results", [])
        steering_results = exp24.get("steering_results", [])

        if not clf_results:
            continue

        found = True

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        # Left: Accuracy by layer
        ax1 = axes[0]
        layers = [r["layer"] for r in clf_results]
        accs = [r["accuracy"] * 100 for r in clf_results]
        cv_accs = [r["cv_accuracy"] * 100 for r in clf_results]

        ax1.plot(layers, accs, "o-", label="Test accuracy", color="#2ecc71", linewidth=2)
        ax1.plot(layers, cv_accs, "s--", label="CV accuracy", color="#3498db", linewidth=2)

        # Add random baseline
        random_baseline = exp24.get("random_baseline", {})
        if random_baseline:
            rand_mean = random_baseline.get("mean", 0) * 100
            ax1.axhline(y=rand_mean, color="gray", linestyle=":", label="Random baseline")

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Language classification accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: Steering prediction shift
        ax2 = axes[1]
        if steering_results:
            pairs = [f"{r['source_lang']}->{r['target_lang']}" for r in steering_results]
            shifts = [r["prediction_shift"] * 100 for r in steering_results]

            y = np.arange(len(pairs))
            colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in shifts]
            ax2.barh(y, shifts, color=colors, edgecolor="black")
            ax2.axvline(x=0, color="black", linewidth=1)
            ax2.set_yticks(y)
            ax2.set_yticklabels(pairs)
            ax2.set_xlabel("Prediction shift (%)")
            ax2.set_title("Steering shifts classifier predictions")
        else:
            ax2.text(0.5, 0.5, "No steering data", ha="center", va="center",
                    transform=ax2.transAxes)

        fig.suptitle(f"SAE-based language detector (Exp24; {label})", fontsize=14)
        plt.tight_layout()

        output_path = output_dir / f"fig_exp24_sae_detector{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp24_sae_detector results found")


def plot_family_causal_ablation(results, output_dir):
    """Figure: Family feature causal ablation (Exp25).

    Shows impact of ablating Indo-Aryan vs Dravidian features on steering.
    """
    if not HAS_MATPLOTLIB:
        return

    found = False
    for label, suffix, exp25 in iter_variants(results, "exp25_family_causal"):
        if not exp25:
            continue

        ablation_results = exp25.get("ablation_results", [])
        causal = exp25.get("causal_analysis", {})

        if not ablation_results:
            continue

        found = True

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        # Left: Ablation impact on script accuracy
        ax1 = axes[0]

        pairs = []
        baseline = []
        ablate_src = []
        ablate_tgt = []

        for r in ablation_results:
            pair = f"{r['source_lang']}->{r['target_lang']}"
            pairs.append(pair)
            baseline.append(r["baseline_script_accuracy"] * 100)
            ablate_src.append(r["ablate_source_script_acc"] * 100)
            ablate_tgt.append(r["ablate_target_script_acc"] * 100)

        x = np.arange(len(pairs))
        width = 0.25

        ax1.bar(x - width, baseline, width, label="Baseline", color="#2ecc71")
        ax1.bar(x, ablate_src, width, label="Ablate source family", color="#e74c3c")
        ax1.bar(x + width, ablate_tgt, width, label="Ablate target family", color="#3498db")

        ax1.set_xticks(x)
        ax1.set_xticklabels(pairs, rotation=45, ha="right")
        ax1.set_ylabel("Script accuracy (%)")
        ax1.set_title("Effect of family ablation on steering")
        ax1.legend(fontsize=8)

        # Right: Causal summary
        ax2 = axes[1]
        ax2.axis("off")

        summary_text = [
            f"Indo-Aryan features necessary for IA steering: {causal.get('ia_features_necessary_for_ia', 'N/A')}",
            f"Dravidian features necessary for DR steering: {causal.get('dr_features_necessary_for_dr', 'N/A')}",
            f"IA features sufficient for DR steering: {causal.get('ia_features_sufficient_for_dr', 'N/A')}",
            f"DR features sufficient for IA steering: {causal.get('dr_features_sufficient_for_ia', 'N/A')}",
            "",
            f"Family separation is causal: {causal.get('family_separation_causal', 'N/A')}",
        ]

        for i, line in enumerate(summary_text):
            ax2.text(0.1, 0.85 - i * 0.12, line, fontsize=10, transform=ax2.transAxes,
                    verticalalignment="top")

        ax2.set_title("Causal analysis summary")

        fig.suptitle(f"Family feature causality (Exp25; {label})", fontsize=14)
        plt.tight_layout()

        output_path = output_dir / f"fig_exp25_family_causal{suffix}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    if not found:
        print("No exp25_family_causal results found")


def generate_all_plots(results_dir):

    """Generate all publication-quality plots."""

    results_dir = Path(results_dir)

    output_dir = results_dir / "figures"

    output_dir.mkdir(exist_ok=True)

    

    print(f"Loading results from: {results_dir}")

    results = load_results(results_dir)

    

    print(f"Found {len(results)} result files: {list(results.keys())}")

    print(f"Saving figures to: {output_dir}")

    print()

    

    # Generate each plot

    # Representation geometry / clustering
    plot_exp1_selectivity_heatmap(results, output_dir)
    plot_hindi_urdu_overlap(results, output_dir)
    plot_script_semantics_controls(results, output_dir)
    plot_language_overlap_heatmaps(results, output_dir)
    plot_hierarchy_profile_similarity(results, output_dir)
    plot_language_agnostic_alignment(results, output_dir)

    # Steering / spillover (only produced when Exp4/9 ran)
    plot_steering_profile_similarity(results, output_dir)
    plot_exp9_layer_method_heatmaps(results, output_dir)
    plot_spillover_analysis(results, output_dir)

    # Robustness / judge calibration (only produced when Exp11/12 ran)
    plot_exp11_judge_calibration(results, output_dir)
    plot_exp12_qa_deltas(results, output_dir)

    # Qualitative / appendix-style visuals (safe to skip)
    plot_feature_labels_summary(results, output_dir)

    # New experiments: Indic cluster validation (Exp18-21)
    plot_exp18_typological_features(results, output_dir)
    plot_exp19_causal_profile(results, output_dir)
    plot_exp20_frequency_control(results, output_dir)
    plot_exp21_family_separation(results, output_dir)

    # Publication-grade experiments (Exp22-25)
    plot_family_transfer_matrix(results, output_dir)
    plot_monosemanticity_distribution(results, output_dir)
    plot_hierarchy_causal_profile(results, output_dir)
    plot_sae_detector_analysis(results, output_dir)
    plot_family_causal_ablation(results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality plots')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing JSON result files')
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib seaborn")
        return
    
    generate_all_plots(args.results_dir)


if __name__ == "__main__":
    main()
