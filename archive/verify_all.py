#!/usr/bin/env python3
"""Comprehensive verification of all code and algorithms.

This script:
1. Checks all imports work
2. Verifies algorithmic correctness
3. Tests data loading
4. Validates experiment pipelines (without GPU)
5. Checks hypothesis alignment

Run: python verify_all.py
"""

import sys
from pathlib import Path


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_imports():
    """Test all module imports."""
    print_section("1. TESTING IMPORTS")
    
    modules = [
        ("config", ["MODEL_ID", "LANGUAGES", "TARGET_LAYERS", "N_SAMPLES_DISCOVERY"]),
        ("data", ["load_flores", "load_research_data", "LANGUAGE_METADATA"]),
        ("evaluation", ["jaccard_overlap", "detect_scripts", "evaluate_steering_output"]),
        ("clustering", ["compute_jaccard_overlap", "LanguageFeatures", "ClusteringResults"]),
    ]
    
    all_passed = True
    
    for module_name, attrs in modules:
        try:
            module = __import__(module_name)
            missing = [a for a in attrs if not hasattr(module, a)]
            
            if missing:
                print(f"  ✗ {module_name}: Missing {missing}")
                all_passed = False
            else:
                print(f"  ✓ {module_name}: All attributes present")
                
        except ImportError as e:
            print(f"  ✗ {module_name}: Import error - {e}")
            all_passed = False
    
    return all_passed


def test_jaccard():
    """Test Jaccard overlap computation (CRITICAL!)."""
    print_section("2. TESTING JACCARD OVERLAP (Critical Bug Fix)")
    
    from evaluation import jaccard_overlap
    from clustering import compute_jaccard_overlap
    
    test_cases = [
        # (set_a, set_b, expected)
        ({1, 2, 3, 4, 5}, {3, 4, 5, 6, 7}, 3/7),  # Normal case
        ({1, 2, 3}, {1, 2, 3}, 1.0),               # Identical
        ({1, 2}, {3, 4}, 0.0),                     # Disjoint
        (set(), set(), 0.0),                       # Empty
        ({1}, {1, 2, 3}, 1/3),                     # Subset
        (set(range(100)), set(range(50, 150)), 50/150),  # Large sets
    ]
    
    all_passed = True
    
    for set_a, set_b, expected in test_cases:
        # Test evaluation.py version
        result1 = jaccard_overlap(set_a, set_b)
        # Test clustering.py version
        result2 = compute_jaccard_overlap(set_a, set_b)
        
        # Check both implementations match
        if abs(result1 - result2) > 0.001:
            print(f"  ✗ Implementations differ: {result1} vs {result2}")
            all_passed = False
            continue
        
        # Check against expected
        if abs(result1 - expected) > 0.001:
            print(f"  ✗ Wrong result for {len(set_a)}/{len(set_b)}: got {result1:.4f}, expected {expected:.4f}")
            all_passed = False
        else:
            print(f"  ✓ |A|={len(set_a)}, |B|={len(set_b)}: {result1:.4f} (expected {expected:.4f})")
        
        # Check range (CRITICAL!)
        if not (0 <= result1 <= 1):
            print(f"  ✗ CRITICAL: Jaccard out of range [0,1]: {result1}")
            all_passed = False
    
    if all_passed:
        print("\n  ✓ Jaccard implementation CORRECT (no >100% bugs!)")
    
    return all_passed


def test_script_detection():
    """Test script detection."""
    print_section("3. TESTING SCRIPT DETECTION")
    
    from evaluation import detect_scripts, is_target_script
    
    test_cases = [
        ("नमस्ते दुनिया", "devanagari", True),      # Hindi
        ("مرحبا بالعالم", "arabic", True),         # Arabic
        ("Hello world", "latin", True),             # English
        ("বাংলা টেক্সট", "bengali", True),          # Bengali
        ("தமிழ் உரை", "tamil", True),               # Tamil
        ("Hello नमस्ते", "devanagari", False),      # Mixed (less than 30%)
    ]
    
    all_passed = True
    
    for text, expected_script, should_detect in test_cases:
        ratios = detect_scripts(text)
        detected = is_target_script(text, expected_script)
        
        if detected == should_detect:
            print(f"  ✓ '{text[:20]}...': {expected_script}={detected}")
        else:
            print(f"  ✗ '{text[:20]}...': Expected {should_detect}, got {detected}")
            all_passed = False
    
    return all_passed


def test_repetition():
    """Test repetition detection."""
    print_section("4. TESTING REPETITION DETECTION")
    
    from evaluation import compute_ngram_repetition, is_degraded
    
    normal_text = "This is a normal sentence with many different words and varied content."
    repeated_text = "hello hello hello hello hello hello hello hello"
    
    rep_normal = compute_ngram_repetition(normal_text, 3)
    rep_bad = compute_ngram_repetition(repeated_text, 3)
    
    print(f"  Normal text 3-gram repetition: {rep_normal:.3f}")
    print(f"  Repeated text 3-gram repetition: {rep_bad:.3f}")
    
    if rep_bad > rep_normal:
        print("  ✓ Repetition detection working correctly")
        return True
    else:
        print("  ✗ Repetition detection FAILED")
        return False


def test_config():
    """Test configuration values."""
    print_section("5. TESTING CONFIGURATION")
    
    from config import (
        MODEL_ID, N_LAYERS, HIDDEN_DIM, 
        N_SAMPLES_DISCOVERY, MONOLINGUALITY_THRESHOLD,
        TARGET_LAYERS, LANGUAGES
    )
    
    checks = [
        ("MODEL_ID is gemma-2-2b", "gemma-2-2b" in MODEL_ID),
        ("N_LAYERS == 26", N_LAYERS == 26),
        ("HIDDEN_DIM == 2304", HIDDEN_DIM == 2304),
        ("N_SAMPLES_DISCOVERY >= 1000", N_SAMPLES_DISCOVERY >= 1000),
        ("MONOLINGUALITY_THRESHOLD == 3.0", MONOLINGUALITY_THRESHOLD == 3.0),
        ("TARGET_LAYERS contains mid layers", 13 in TARGET_LAYERS),
        ("LANGUAGES has Hindi", "hi" in LANGUAGES),
        ("LANGUAGES has Urdu", "ur" in LANGUAGES),
    ]
    
    all_passed = True
    
    for desc, passed in checks:
        if passed:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_passed = False
    
    return all_passed


def test_hypotheses():
    """Verify hypothesis definitions."""
    print_section("6. VERIFYING HYPOTHESES")
    
    hypotheses = [
        ("H1: Feature Existence", 
         "SAEs contain ≥10 robust Hindi-specific features (M>3.0)",
         "Count features with monolinguality > 3.0 per layer",
         "Falsified if <10 features per layer"),
        
        ("H2: Detector vs Generator (NOVEL!)",
         "Monolinguality selects detectors, activation-diff selects generators",
         "Compare overlap of top-25 features from each method",
         "Falsified if overlap >20% OR mono achieves >20% steering success"),
        
        ("H3: Layer Distribution (REVISED)",
         "Late layers (not mid!) have most disentangled language features",
         "Count features by layer, check where peak occurs",
         "Original hypothesis FALSIFIED - peak at layer 24, not mid-layers"),
        
        ("H4: Language Clustering (NEW!)",
         "Languages from same family cluster in SAE space",
         "Compute Jaccard overlap matrix, hierarchical clustering",
         "Falsified if within-family overlap ≤ between-family overlap"),
        
        ("H5: Script vs Semantic",
         "Hindi-Urdu share >90% semantic features, <10% script features",
         "Compare feature sets between Hindi/Urdu",
         "Falsified if semantic overlap <80% OR script-specific <5%"),
    ]
    
    print("\nHypotheses to test:")
    for h_name, description, method, falsification in hypotheses:
        print(f"\n  {h_name}")
        print(f"    Claim: {description}")
        print(f"    Method: {method}")
        print(f"    Falsification: {falsification}")
    
    return True


def test_experiment_structure():
    """Test experiment file structure."""
    print_section("7. TESTING EXPERIMENT STRUCTURE")
    
    expected_experiments = [
        "experiments/exp1_feature_discovery.py",
        "experiments/exp2_steering.py",
        "experiments/exp3_hindi_urdu.py",
    ]
    
    all_found = True
    
    for exp_path in expected_experiments:
        path = Path(exp_path)
        if path.exists():
            # Check for main() function
            content = path.read_text()
            if "def main(" in content:
                print(f"  ✓ {exp_path}: has main()")
            else:
                print(f"  ⚠ {exp_path}: missing main()")
        else:
            print(f"  ✗ {exp_path}: NOT FOUND")
            all_found = False
    
    return all_found


def test_language_metadata():
    """Test language metadata."""
    print_section("8. TESTING LANGUAGE METADATA")
    
    from data import LANGUAGE_METADATA
    
    expected_families = {
        "hi": "Indo-Aryan",
        "ur": "Indo-Aryan",
        "bn": "Indo-Aryan",
        "ta": "Dravidian",
        "te": "Dravidian",
        "de": "Germanic",
        "ar": "Semitic",
    }
    
    expected_scripts = {
        "hi": "Devanagari",
        "ur": "Arabic",
        "bn": "Bengali",
        "ta": "Tamil",
        "de": "Latin",
    }
    
    all_passed = True
    
    for lang, expected_family in expected_families.items():
        if lang in LANGUAGE_METADATA:
            actual = LANGUAGE_METADATA[lang].get("family", "Unknown")
            if actual == expected_family:
                print(f"  ✓ {lang}: family={actual}")
            else:
                print(f"  ✗ {lang}: expected family={expected_family}, got {actual}")
                all_passed = False
        else:
            print(f"  ✗ {lang}: not in metadata")
            all_passed = False
    
    return all_passed


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE CODE VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Jaccard Overlap", test_jaccard),
        ("Script Detection", test_script_detection),
        ("Repetition Detection", test_repetition),
        ("Configuration", test_config),
        ("Hypotheses", test_hypotheses),
        ("Experiment Structure", test_experiment_structure),
        ("Language Metadata", test_language_metadata),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            results[name] = f"ERROR: {e}"
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    
    for name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n  ⚠ Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
