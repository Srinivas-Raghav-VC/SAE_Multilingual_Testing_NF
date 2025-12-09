@echo off
echo ===================================================
echo Running SAE Multilingual Steering Experiments
echo ===================================================

echo 1. Judge Calibration (Exp11) - Critical for Exp9/12
python experiments/exp11_judge_calibration.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo 2. Feature Discovery (Exp1)
python experiments/exp1_feature_discovery.py

echo 3. Steering Comparison (Exp2)
python experiments/exp2_steering.py

echo 4. Hindi-Urdu Overlap (Exp3)
python experiments/exp3_hindi_urdu_fixed.py

echo 5. Spillover (Exp4)
python experiments/exp4_spillover.py

echo 6. Hierarchical Analysis (Exp5)
python experiments/exp5_hierarchical.py

echo 7. Script vs Semantic (Exp6)
python experiments/exp6_script_semantics_controls.py

echo 8. Causal Probing (Exp7)
python experiments/exp7_causal_feature_probing.py

echo 9. Layer Sweep (Exp9)
python experiments/exp9_layer_sweep_steering.py

echo 10. QA Degradation (Exp12)
python experiments/exp12_qa_degradation.py

echo 11. Cross-Lingual Alignment (Exp14)
python experiments/exp14_language_agnostic_space.py

echo 12. Directional Symmetry (Exp15)
python experiments/exp15_directional_symmetry.py

echo 13. Code-Mix Robustness (Exp16)
python experiments/exp16_code_mix_robustness.py

echo ===================================================
echo Generating Plots and Summary
echo ===================================================
python summarize_results.py
python plots.py

echo Done!
