# Figures for SAE Multilingual Steering Paper

This directory should contain the following figures for the paper:

## Required Conceptual Figures (to be created manually)

1. **architecture_multilingual_sae.{pdf,png}**
   - Gemma-2 with SAE attachment points
   - Shows: multilingual inputs → transformer layers → SAE encoding/decoding → steering vectors
   - Referenced in: Figure 1

2. **steering_eval_pipeline.{pdf,png}**
   - Complete evaluation pipeline diagram
   - Shows: steering methods → generation → metrics (script, LaBSE, degradation) → calibrated judge
   - Referenced in: Figure 2

3. **overview_comic.{pdf,png}**
   - 4-panel summary of the approach
   - Panel 1: Multilingual prompts
   - Panel 2: SAE decomposition + layer structure
   - Panel 3: Steering strength effect
   - Panel 4: Evaluation metrics
   - Referenced in: Figure 3

4. **language_concept_space.{pdf,png}**
   - Conceptual hierarchy diagram
   - Shows: early (orthography) → mid (shared semantics) → late (language-specific decoding)
   - Referenced in: Figure 4

## Auto-Generated Figures (created by plots.py)

These will be created in `results/figures/` after running experiments:

- `fig1_feature_counts_by_layer.{pdf,png}` - Monolinguality detector counts per layer
- `fig2_steering_comparison.{pdf,png}` - Method comparison
- `fig3_jaccard_overlaps.{pdf,png}` - Hindi-Urdu vs Hindi-English Jaccard
- `fig4_spillover_matrix.{pdf,png}` - Language distribution under steering
- `fig5_qa_degradation.{pdf,png}` - QA performance baseline vs steered

## Creation Tools

Recommended tools for creating conceptual figures:
- draw.io / diagrams.net (free, exports to PDF)
- Inkscape (free, vector graphics)
- Lucidchart
- Figma

## Notes

- All figures should be vector format (PDF preferred) for print quality
- Use consistent color scheme across figures
- Include figure captions in the paper that fully describe the content
