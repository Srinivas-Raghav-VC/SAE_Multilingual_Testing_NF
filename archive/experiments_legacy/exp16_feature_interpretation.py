"""Experiment 16: Automated Feature Interpretation

Addresses the "Mechanistic Depth" gap by inspecting the specific SAE features
that drive the Indic steering vector.

Workflow:
1. Identify top-k features in the EN->HI steering vector.
2. Find max-activating examples for these features in Hindi, Urdu, Bengali, and English.
3. Use Gemini to generate a natural language explanation of what the feature detects.
4. Output a "Feature Dashboard" Markdown report with cross-lingual analysis.

Key Question: Are the top steering features detecting:
- Script-specific patterns (Devanagari shapes)?
- Language-specific vocabulary (Hindi words)?
- Cross-lingual semantic concepts (shared meaning across Indic languages)?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from typing import List, Dict, Tuple
from tqdm import tqdm

from config import TARGET_LAYERS, EVAL_PROMPTS, GOOGLE_API_KEY
from model import GemmaWithSAE
from data import load_research_data
from steering_utils import get_activation_diff_features
from evaluation_comprehensive import _respect_gemini_rpm, is_gemini_available

# Prompt for Auto-Interpretation
INTERP_PROMPT = """You are an expert linguist and mechanistic interpretability researcher.
I will show you a list of text examples that highly activate a specific neuron (feature) in a language model.
The examples may be in Hindi, Urdu, or English.

Activating Examples:
{examples}

Your Goal:
Identify the *precise* linguistic or semantic concept this feature detects.
Is it:
- A specific token or subword?
- A grammatical structure (e.g., relative clauses)?
- A semantic topic (e.g., family, politics, geography)?
- A script-specific pattern?

Format:
Return a JSON object with:
- "short_name": A 3-5 word label (e.g., "Hindi Verbs", "Urdu/Hindi Political Terms").
- "description": A 1-2 sentence technical explanation.
- "is_multilingual": boolean (true if it activates for meaning across languages, false if script-specific).
"""

def get_max_activating_examples(
    model: GemmaWithSAE,
    layer: int,
    feature_idx: int,
    dataset: List[str],
    k: int = 5
) -> List[Tuple[str, float]]:
    """Find the k sentences with highest activation for a specific feature."""
    activations = []
    
    # Batch processing would be faster, but keeping it simple/safe for memory first
    # We scan a subset of the dataset to be reasonable
    scan_limit = 500
    
    sae = model.load_sae(layer)
    
    for text in dataset[:scan_limit]:
        # We need the max activation over the sequence length
        try:
            # sae_acts: [seq_len, d_sae]
            acts = model.get_sae_activations(text, layer)
            # Get max activation for this feature in this sentence
            val = acts[:, feature_idx].max().item()
            if val > 0.1: # Only keep if non-trivial
                activations.append((text, val))
        except Exception:
            continue
            
    # Sort by activation value (descending)
    activations.sort(key=lambda x: x[1], reverse=True)
    return activations[:k]

def generate_interpretation(
    feature_id: int,
    examples: List[Tuple[str, float, str]],  # (text, score, lang)
    api_key: str
) -> Dict:
    """Call Gemini to interpret the feature."""
    if not api_key:
        return {"short_name": "No API Key", "description": "Skipped interpretation.", "is_multilingual": False}

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Format examples for the prompt
    ex_text = ""
    for text, score, lang in examples:
        ex_text += f"- [{lang.upper()}] (Act: {score:.2f}): {text}\n"
        
    try:
        _respect_gemini_rpm(60) # Reuse rate limiter from eval
        response = model.generate_content(
            INTERP_PROMPT.format(examples=ex_text),
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Interpretation failed for feature {feature_id}: {e}")
        return {"short_name": "Error", "description": str(e), "is_multilingual": False}

def main():
    print("=" * 60)
    print("EXPERIMENT 16: Automated Feature Interpretation")
    print("=" * 60)
    
    if not is_gemini_available():
        print("WARNING: Gemini API not available. Auto-interp will be skipped.")
        
    # 1. Setup
    model = GemmaWithSAE()
    model.load_model()
    
    # Use a mid-late layer where semantic abstraction is high
    # Layer 20 for 2B is standard in this repo
    layer = 20 if 20 in TARGET_LAYERS else TARGET_LAYERS[len(TARGET_LAYERS)//2]
    
    print(f"\nAnalyzing Layer {layer}...")

    # 2. Data
    data = load_research_data(max_train_samples=500)
    
    # 3. Identify Top Features (Steering Vector Components)
    # Re-run the logic that selects features for EN->HI steering
    print("Identifying top steering features (EN->HI)...")
    
    # Calculate difference
    # Note: We duplicate some logic from steering_utils to get specific values
    sae = model.load_sae(layer)
    
    # Get mean activations
    print("  Computing mean activations...")
    
    def get_mean_acts(texts):
        acts_list = []
        for t in texts[:100]:
            a = model.get_sae_activations(t, layer)
            acts_list.append(a.mean(dim=0))
        return torch.stack(acts_list).mean(dim=0)

    hi_mean = get_mean_acts(data.train["hi"])
    en_mean = get_mean_acts(data.train["en"])
    
    diff = hi_mean - en_mean
    
    # Get top 10 positive features (drives Hindi)
    values, indices = diff.topk(10)
    top_features = indices.tolist()
    top_scores = values.tolist()
    
    print(f"  Top 10 features: {top_features}")
    
    # 4. Interpret Features
    report_data = []

    # Determine model variant for output naming
    use_9b = "9b" in str(getattr(model, "model_id", "")).lower()
    suffix = "_9b" if use_9b else ""

    # We will look for examples in HI, UR, BN, and EN to test "Indic Cluster" hypothesis
    # Including Bengali tests whether features generalize beyond Hindi-Urdu
    search_corpus = {
        "hi": data.train.get("hi", []),
        "ur": data.train.get("ur", []),
        "bn": data.train.get("bn", []),
        "en": data.train.get("en", []),
    }

    # Warn about missing languages
    missing = [lang for lang, texts in search_corpus.items() if not texts]
    if missing:
        print(f"  Warning: No data for languages: {missing}")
    
    print("\nGenerating interpretations...")
    for rank, (feat_idx, score) in enumerate(zip(top_features, top_scores)):
        print(f"  Processing Feature {feat_idx} (Rank {rank+1}, Score {score:.4f})...")
        
        # Collect max activating examples from each language
        feature_examples = []
        
        for lang, texts in search_corpus.items():
            if not texts: continue
            top_k = get_max_activating_examples(model, layer, feat_idx, texts, k=3)
            for text, act in top_k:
                feature_examples.append((text, act, lang))
        
        # Sort combined examples by activation
        feature_examples.sort(key=lambda x: x[1], reverse=True)
        top_examples = feature_examples[:10] # Show top 10 overall to LLM
        
        # Auto-Interpret
        interpretation = generate_interpretation(
            feat_idx, 
            top_examples, 
            GOOGLE_API_KEY
        )
        
        entry = {
            "rank": rank + 1,
            "feature_id": feat_idx,
            "steering_score": score,
            "interpretation": interpretation,
            "top_examples": [
                {"text": t, "activation": a, "lang": l} 
                for t, a, l in top_examples[:5] # Save top 5
            ]
        }
        report_data.append(entry)

    # 5. Compute Summary Statistics
    n_multilingual = sum(1 for r in report_data if r["interpretation"].get("is_multilingual", False))
    n_script_specific = len(report_data) - n_multilingual

    summary = {
        "layer": layer,
        "n_features_analyzed": len(report_data),
        "n_multilingual_semantic": n_multilingual,
        "n_script_or_lang_specific": n_script_specific,
        "multilingual_ratio": n_multilingual / len(report_data) if report_data else 0,
        "interpretation": (
            "Majority are cross-lingual semantic features" if n_multilingual > n_script_specific
            else "Majority are script/language-specific features"
        ),
    }

    # 6. Generate Markdown Report
    report_path = Path("results") / f"exp16_feature_interpretation{suffix}.md"
    json_path = Path("results") / f"exp16_feature_interpretation{suffix}.json"

    output_data = {
        "summary": summary,
        "features": report_data,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment 16: Top Feature Interpretation (Layer {layer})\n\n")
        f.write("Analysis of the top 10 SAE features driving the EN->HI steering vector.\n\n")

        # Summary box
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Features Analyzed | {summary['n_features_analyzed']} |\n")
        f.write(f"| Multilingual (Semantic) | {summary['n_multilingual_semantic']} |\n")
        f.write(f"| Script/Language-Specific | {summary['n_script_or_lang_specific']} |\n")
        f.write(f"| Multilingual Ratio | {summary['multilingual_ratio']:.0%} |\n\n")
        f.write(f"**Interpretation:** {summary['interpretation']}\n\n")
        f.write("---\n\n")

        for item in report_data:
            interp = item["interpretation"]
            f.write(f"## {item['rank']}. Feature {item['feature_id']}: {interp.get('short_name', 'Unknown')}\n\n")
            f.write(f"**Steering Score:** {item['steering_score']:.4f}\n\n")
            f.write(f"**AI Description:** {interp.get('description', 'N/A')}\n\n")
            f.write(f"**Multilingual (Semantic)?** {'✓ Yes' if interp.get('is_multilingual') else '✗ No (Script/Lang specific)'}\n\n")
            f.write("**Top Activating Examples:**\n")
            for ex in item["top_examples"]:
                clean_text = ex['text'].replace('\n', ' ').strip()[:100]
                f.write(f"- **[{ex['lang'].upper()}]** ({ex['activation']:.2f}): {clean_text}...\n")
            f.write("\n---\n\n")

    print(f"\n✓ Report saved to {report_path}")
    print(f"✓ Data saved to {json_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print("FEATURE INTERPRETATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Multilingual (semantic): {n_multilingual}/{len(report_data)}")
    print(f"  Script/lang-specific:    {n_script_specific}/{len(report_data)}")
    print(f"  → {summary['interpretation']}")

if __name__ == "__main__":
    main()
