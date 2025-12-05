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


def plot_steering_comparison(results, output_dir):
    """Figure 2: Steering method comparison.
    
    Line plot showing success rate vs steering strength for each method.
    """
    if not HAS_MATPLOTLIB:
        return
    
    exp2 = results.get("exp2_steering_comparison", {})
    if not exp2:
        print("No exp2 results found")
        return
    
    layers = exp2.get("layers", {})
    if not layers:
        print("exp2_steering_comparison has no 'layers' key")
        return
    
    # Use layer 13 if present, else first available
    layer_key = "13" if "13" in layers else sorted(layers.keys(), key=int)[0]
    methods = layers[layer_key].get("methods", {})
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    colors = {'activation_diff': '#3498db', 'monolinguality': '#e74c3c', 
              'random': '#95a5a6', 'dense': '#9b59b6'}
    markers = {'activation_diff': 'o', 'monolinguality': 's', 
               'random': '^', 'dense': 'D'}
    
    for method, data in methods.items():
        strengths = sorted(float(s) for s in data.keys())
        success_rates = [data[str(s)]["success_rate"] * 100 for s in strengths]
        
        ax.plot(strengths, success_rates, 
                marker=markers.get(method, 'o'),
                color=colors.get(method, 'gray'),
                label=method.replace('_', ' ').title(),
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Steering Strength')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Steering Method Comparison')
    ax.set_ylim(-5, 105)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / "fig2_steering_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hindi_urdu_overlap(results, output_dir):
    """Figure 3: Hindi-Urdu feature overlap across layers.
    
    Line plot showing Jaccard overlap between Hindi and Urdu features.
    """
    if not HAS_MATPLOTLIB:
        return
    
    exp3 = results.get("exp3_hindi_urdu_fixed", {})
    if not exp3:
        print("No exp3 results found")
        return
    
    layers_data = exp3.get("layers", {})
    if not layers_data:
        print("exp3_hindi_urdu_fixed has no 'layers' key")
        return
    
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
    ax1.plot(layers, hi_ur_overlap, 'o-', color='#2ecc71', 
             label='Hindi-Urdu', linewidth=2, markersize=8)
    ax1.plot(layers, hi_en_overlap, 's-', color='#3498db', 
             label='Hindi-English', linewidth=2, markersize=8)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='H4 threshold')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Jaccard Overlap (%)')
    ax1.set_title('Feature Overlap Across Layers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 110)
    
    # Right: Script-specific features
    x = np.arange(len(layers))
    width = 0.35
    
    ax2.bar(x - width/2, script_features_hi, width, label='Hindi (Devanagari)', color='#e74c3c')
    ax2.bar(x + width/2, script_features_ur, width, label='Urdu (Arabic)', color='#9b59b6')
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Script-Specific Feature Count')
    ax2.set_title('Script-Only Features by Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.legend()
    
    plt.tight_layout()
    output_path = output_dir / "fig3_hindi_urdu_overlap.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_script_semantics_controls(results, output_dir):
    """Figure 4: Script vs semantic feature counts by layer (Exp6).
    
    Uses exp6_script_semantics_controls.json to show how many features
    are script-only vs semantic vs family-semantic across layers.
    """
    if not HAS_MATPLOTLIB:
        return
    
    exp6 = results.get("exp6_script_semantics_controls", {})
    if not exp6:
        print("No exp6 results found")
        return
    
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
    ax.set_title("Script vs Semantic Features Across Layers (Exp6)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    
    output_path = output_dir / "fig4_script_semantics_controls.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_monolinguality_distribution(results, output_dir):
    """Figure 5: Distribution of monolinguality scores.
    
    Histogram showing how monolinguality scores are distributed.
    """
    if not HAS_MATPLOTLIB:
        return
    
    exp1 = results.get("exp1_feature_discovery", {})
    if not exp1:
        print("No exp1 results found")
        return
    
    # This requires the raw monolinguality scores, which may not be in JSON
    # Create placeholder plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # Placeholder: Show threshold lines
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='M=1.0')
    ax.axvline(x=3.0, color='red', linestyle='-', alpha=0.8, label='M=3.0 (threshold)')
    ax.axvline(x=5.0, color='green', linestyle='--', alpha=0.5, label='M=5.0')
    
    ax.set_xlabel('Monolinguality Score (M)')
    ax.set_ylabel('Feature Count')
    ax.set_title('Monolinguality Score Distribution\n(Requires raw data - placeholder)')
    ax.legend()
    ax.set_xlim(0, 10)
    
    output_path = output_dir / "fig4_monolinguality_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


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
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # Table data
    columns = ['Hypothesis', 'Prediction', 'Result', 'Status']
    data = [
        ['H1', '≥10 Hindi features/layer', '32-96 features', '✓ PASS'],
        ['H2', 'Monolinguality ≥ Act-diff', 'Act-diff: 100%, Mono: 0%', '⚠ ANOMALY'],
        ['H3', 'Peak in mid-layers (10-16)', 'Peak at layer 24', '✗ FAIL'],
        ['H4', 'Hindi-Urdu >50% overlap', '94-99% overlap', '✓ PASS'],
    ]
    
    # Colors for status
    cell_colors = []
    for row in data:
        row_colors = ['white', 'white', 'white', 'white']
        if '✓' in row[3]:
            row_colors[3] = '#d4edda'  # Green
        elif '✗' in row[3]:
            row_colors[3] = '#f8d7da'  # Red
        else:
            row_colors[3] = '#fff3cd'  # Yellow
        cell_colors.append(row_colors)
    
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=['#e9ecef'] * 4,
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    ax.set_title('Summary of Hypothesis Testing Results', fontsize=14, fontweight='bold', y=0.85)
    
    output_path = output_dir / "table1_hypothesis_summary.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


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
    
    exp5 = results.get("exp5_hierarchical_analysis", {})
    if not exp5:
        print("No exp5_hierarchical_analysis results found")
        return
    
    analyses = exp5.get("layer_analyses", [])
    if not analyses:
        print("No layer_analyses in exp5_hierarchical_analysis")
        return
    
    # Collect language codes from overlap keys like "hi-ur"
    lang_set = set()
    for entry in analyses:
        overlaps = entry.get("overlaps", {})
        for key in overlaps.keys():
            parts = key.split("-")
            if len(parts) == 2:
                lang_set.update(parts)
    if not lang_set:
        print("No overlap entries with language codes found")
        return
    
    langs = sorted(lang_set)
    lang_idx = {l: i for i, l in enumerate(langs)}
    
    # Helper to accumulate overlaps by group
    groups = {
        "early": {"layers": [], "acc": np.zeros((len(langs), len(langs)))},
        "mid": {"layers": [], "acc": np.zeros((len(langs), len(langs)))},
        "late": {"layers": [], "acc": np.zeros((len(langs), len(langs)))},
    }
    
    for entry in analyses:
        layer = entry.get("layer", None)
        if layer is None:
            continue
        overlaps = entry.get("overlaps", {})
        
        if layer <= 8:
            group = "early"
        elif 10 <= layer <= 16:
            group = "mid"
        else:
            group = "late"
        
        groups[group]["layers"].append(layer)
        mat = groups[group]["acc"]
        
        for key, val in overlaps.items():
            parts = key.split("-")
            if len(parts) != 2:
                continue
            a, b = parts
            if a in lang_idx and b in lang_idx:
                i, j = lang_idx[a], lang_idx[b]
                mat[i, j] += val
    
    # Create heatmaps for each group where we have layers
    for group_name, data in groups.items():
        layers = data["layers"]
        if not layers:
            continue
        mat = data["acc"] / max(len(layers), 1)
        
        fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
        sns.heatmap(
            mat,
            xticklabels=langs,
            yticklabels=langs,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            annot=False,
            ax=ax,
        )
        ax.set_title(f"Language Jaccard Overlap ({group_name} layers)")
        ax.set_xlabel("Language")
        ax.set_ylabel("Language")
        
        output_path = output_dir / f"fig8_language_overlap_{group_name}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


def plot_steering_profile_similarity(results, output_dir):
    """Figure 9: Steering profile similarity across languages (Exp9).
    
    Uses exp9_layer_sweep_steering.json. For each language, builds a
    vector of best script+semantic success per layer (activation_diff
    method), then computes Pearson correlation between languages and
    plots a heatmap.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return
    
    exp9 = results.get("exp9_layer_sweep_steering", {})
    if not exp9:
        print("No exp9_layer_sweep_steering results found")
        return
    
    langs = sorted(exp9.keys())
    if not langs:
        print("No languages in exp9_layer_sweep_steering")
        return
    
    # Determine common set of layers
    all_layers = set()
    for lang in langs:
        layers_d = exp9[lang].get("layers", {})
        all_layers.update(int(l) for l in layers_d.keys())
    layers = sorted(all_layers)
    if not layers:
        print("No layers in exp9_layer_sweep_steering")
        return
    
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
            for strength_str, metrics in meth.items():
                # Prefer script+semantic if available, else script-only
                ss = metrics.get("success_rate_script_semantic", None)
                if ss is None:
                    ss = metrics.get("success_rate_script", 0.0)
                if ss is None:
                    ss = 0.0
                if ss > best:
                    best = ss
            vec.append(best)
        profiles.append(vec)
        valid_langs.append(lang)
    
    if len(profiles) < 2:
        print("Not enough languages with profiles for correlation")
        return
    
    profiles_arr = np.array(profiles)  # shape: (n_langs, n_layers)
    if np.all(profiles_arr == 0):
        print("All steering profiles are zero; skipping similarity plot")
        return
    
    # Compute correlation matrix
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
    ax.set_title("Steering Profile Similarity (activation_diff; Exp9)")
    ax.set_xlabel("Language")
    ax.set_ylabel("Language")
    
    output_path = output_dir / "fig9_steering_profile_similarity.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hierarchy_profile_similarity(results, output_dir):
    """Figure 10: Hierarchical representation profile similarity (Exp5).
    
    Uses exp5_hierarchical_analysis.json. For each language, builds a
    vector of active-feature counts per layer (from per_language) and
    computes Pearson correlation between languages. This shows whether
    languages share a similar *hierarchical* layer profile.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return
    
    exp5 = results.get("exp5_hierarchical_analysis", {})
    if not exp5:
        print("No exp5_hierarchical_analysis results found")
        return
    
    analyses = exp5.get("layer_analyses", [])
    if not analyses:
        print("No layer_analyses in exp5_hierarchical_analysis")
        return
    
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
        print("No per_language or layer data for hierarchy profiles")
        return
    
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
        print("Not enough languages with non-zero hierarchy profiles")
        return
    
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
    ax.set_title("Hierarchy Profile Similarity (Exp5: per-layer active features)")
    ax.set_xlabel("Language")
    ax.set_ylabel("Language")
    
    output_path = output_dir / "fig10_hierarchy_profile_similarity.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


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
    plot_feature_counts_by_layer(results, output_dir)
    plot_steering_comparison(results, output_dir)
    plot_hindi_urdu_overlap(results, output_dir)
    plot_script_semantics_controls(results, output_dir)
    plot_monolinguality_distribution(results, output_dir)
    plot_dead_features(results, output_dir)
    plot_summary_table(results, output_dir)
    plot_feature_labels_summary(results, output_dir)
    plot_language_overlap_heatmaps(results, output_dir)
    plot_steering_profile_similarity(results, output_dir)
    plot_hierarchy_profile_similarity(results, output_dir)
    
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
