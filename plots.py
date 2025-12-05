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
    
    layers = sorted([int(k.split("_")[1]) for k in exp1.keys() if k.startswith("layer_")])
    
    # Extract Hindi feature counts (M > 3.0)
    hi_counts = []
    for layer in layers:
        layer_data = exp1.get(f"layer_{layer}", {})
        mono_features = layer_data.get("monolinguality_features", {})
        hi_count = mono_features.get("hi", 0)
        if isinstance(hi_count, list):
            hi_count = len(hi_count)
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
    
    methods = exp2.get("methods", {})
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    colors = {'activation_diff': '#3498db', 'monolinguality': '#e74c3c', 
              'random': '#95a5a6', 'dense': '#9b59b6'}
    markers = {'activation_diff': 'o', 'monolinguality': 's', 
               'random': '^', 'dense': 'D'}
    
    for method, data in methods.items():
        strengths = sorted([float(k) for k in data.keys() if isinstance(data[k], dict)])
        success_rates = [data[str(s) if str(s) in data else s]["success_rate"] * 100 
                        for s in strengths]
        
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
    
    exp3 = results.get("exp3_hindi_urdu_overlap", {})
    if not exp3:
        print("No exp3 results found")
        return
    
    layers_data = exp3.get("layers", {})
    if not layers_data:
        # Try alternative structure
        layers_data = {k: v for k, v in exp3.items() if k.startswith("layer_")}
    
    layers = sorted([int(k.split("_")[1]) for k in layers_data.keys()])
    
    hi_ur_overlap = []
    hi_en_overlap = []
    script_features_hi = []
    script_features_ur = []
    
    for layer in layers:
        layer_data = layers_data.get(f"layer_{layer}", {})
        hi_ur_overlap.append(layer_data.get("hindi_urdu_jaccard", 0) * 100)
        hi_en_overlap.append(layer_data.get("hindi_english_jaccard", 0) * 100)
        script_features_hi.append(layer_data.get("hindi_script_features", 0))
        script_features_ur.append(layer_data.get("urdu_script_features", 0))
    
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


def plot_monolinguality_distribution(results, output_dir):
    """Figure 4: Distribution of monolinguality scores.
    
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
    """Figure 5: Dead feature analysis.
    
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
    plot_monolinguality_distribution(results, output_dir)
    plot_dead_features(results, output_dir)
    plot_summary_table(results, output_dir)
    
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
