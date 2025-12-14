@echo off
echo ===================================================
echo Running SAE Multilingual Steering Experiments
echo ===================================================

echo 1. Feature Discovery (Exp1)
python experiments/exp1_feature_discovery.py

echo 2. Hindi-Urdu Overlap (Exp3)
python experiments/exp3_hindi_urdu_fixed.py

echo 3. Hierarchical Analysis (Exp5)
python experiments/exp5_hierarchical.py

echo 4. Judge Calibration (Exp11) - used by Exp9/10/12
python experiments/exp11_judge_calibration.py

echo 5. Scaling Sanity Check (Exp8)
python experiments/exp8_scaling_9b_low_resource.py

echo 6. Layer Sweep (Exp9)
python experiments/exp9_layer_sweep_steering.py

echo 7. Attribution Steering (Exp10)
python experiments/exp10_attribution_occlusion.py

echo 8. QA Degradation (Exp12)
python experiments/exp12_qa_degradation.py

echo 9. Script/Semantic Group Ablation (Exp13)
python experiments/exp13_script_semantic_ablation.py

echo 10. Cross-Lingual Alignment (Exp14)
python experiments/exp14_language_agnostic_space.py

echo ===================================================
echo Generating Plots and Summary
echo ===================================================
python summarize_results.py
python plots.py

echo Done!
