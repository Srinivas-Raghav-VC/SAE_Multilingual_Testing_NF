"""Research Validation Module for SAE Multilingual Steering.

This module provides validation functions to ensure research rigor:
- Configuration validation
- Data quality checks
- Sample size verification
- Script/language mapping consistency

Usage:
    from validation import validate_experiment_setup
    validate_experiment_setup()  # Run before any experiment
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    LANGUAGES,
    LANG_TO_SCRIPT,
    SCRIPT_RANGES,
    TARGET_LAYERS,
    MIN_SAMPLES_PER_LANGUAGE,
    MIN_PROMPTS_STEERING,
    MIN_PROMPTS_CAUSAL_PROBING,
    MIN_FEATURES_FOR_STEERING,
    N_SAMPLES_DISCOVERY,
    N_SAMPLES_EVAL,
    EVAL_PROMPTS,
    SAE_RELEASE,
    SAE_WIDTH,
    MODEL_ID,
)


class ValidationError(Exception):
    """Raised when validation fails critically."""
    pass


class ValidationWarning:
    """Container for validation warnings."""
    def __init__(self, message: str, severity: str = "warning"):
        self.message = message
        self.severity = severity
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.message}"


def validate_config() -> List[ValidationWarning]:
    """Validate configuration settings for research rigor.
    
    Returns:
        List of validation warnings (empty if all checks pass)
    """
    warnings = []
    
    # Check that all languages in LANGUAGES have script mappings
    for lang in LANGUAGES.keys():
        if lang not in LANG_TO_SCRIPT:
            warnings.append(ValidationWarning(
                f"Language '{lang}' in LANGUAGES has no entry in LANG_TO_SCRIPT. "
                "This may cause incorrect script ratio calculations.",
                severity="warning"
            ))
    
    # Check that all scripts in LANG_TO_SCRIPT have detection ranges
    # SCRIPT_RANGES now supports both single-range (tuple) and multi-range (list) formats
    for lang, script in LANG_TO_SCRIPT.items():
        if script not in SCRIPT_RANGES:
            warnings.append(ValidationWarning(
                f"Script '{script}' (for language '{lang}') not in SCRIPT_RANGES. "
                "Script detection will fail for this language.",
                severity="error"
            ))
        else:
            # Validate the script range format
            ranges = SCRIPT_RANGES[script]
            if isinstance(ranges, list):
                # Multi-range format: list of (start, end) tuples
                for i, r in enumerate(ranges):
                    if not (isinstance(r, tuple) and len(r) == 2):
                        warnings.append(ValidationWarning(
                            f"Script '{script}' range {i} is malformed: {r}. "
                            "Expected (start, end) tuple.",
                            severity="error"
                        ))
            elif isinstance(ranges, tuple) and len(ranges) == 2:
                # Legacy single-range format: (start, end)
                pass  # Valid
            else:
                warnings.append(ValidationWarning(
                    f"Script '{script}' has invalid range format: {ranges}. "
                    "Expected list of tuples or single tuple.",
                    severity="error"
                ))
    
    # Check sample size configuration
    if N_SAMPLES_DISCOVERY < MIN_SAMPLES_PER_LANGUAGE * len(LANGUAGES):
        warnings.append(ValidationWarning(
            f"N_SAMPLES_DISCOVERY ({N_SAMPLES_DISCOVERY}) may be too small for "
            f"{len(LANGUAGES)} languages (recommend >= {MIN_SAMPLES_PER_LANGUAGE} per language).",
            severity="warning"
        ))
    
    if N_SAMPLES_EVAL < MIN_PROMPTS_STEERING:
        warnings.append(ValidationWarning(
            f"N_SAMPLES_EVAL ({N_SAMPLES_EVAL}) is below MIN_PROMPTS_STEERING ({MIN_PROMPTS_STEERING}). "
            "Steering evaluation results may be statistically unreliable.",
            severity="warning"
        ))
    
    # Check EVAL_PROMPTS
    if len(EVAL_PROMPTS) < MIN_PROMPTS_STEERING:
        warnings.append(ValidationWarning(
            f"EVAL_PROMPTS has only {len(EVAL_PROMPTS)} prompts (recommend >= {MIN_PROMPTS_STEERING}). "
            "Consider adding more prompts or using FLORES augmentation.",
            severity="warning"
        ))
    
    # Validate TARGET_LAYERS
    if not TARGET_LAYERS:
        warnings.append(ValidationWarning(
            "TARGET_LAYERS is empty. No layers will be analyzed.",
            severity="error"
        ))
    elif max(TARGET_LAYERS) >= 26:  # Gemma 2B has 26 layers (0-25)
        warnings.append(ValidationWarning(
            f"TARGET_LAYERS contains layer {max(TARGET_LAYERS)} which may exceed model depth. "
            "Gemma 2B has layers 0-25.",
            severity="warning"
        ))
    
    return warnings


def validate_data_split(train: Dict[str, List], test: Dict[str, List]) -> List[ValidationWarning]:
    """Validate train/test data split for research rigor.
    
    Args:
        train: Training data dict {lang: [sentences]}
        test: Test data dict {lang: [sentences]}
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check for data leakage
    for lang in train.keys():
        if lang in test:
            train_set = set(train[lang])
            test_set = set(test[lang])
            overlap = train_set & test_set
            if overlap:
                warnings.append(ValidationWarning(
                    f"DATA LEAKAGE: {len(overlap)} sentences overlap between train and test for '{lang}'. "
                    "This will invalidate evaluation results!",
                    severity="error"
                ))
    
    # Check minimum sample sizes
    for lang, texts in train.items():
        if len(texts) < MIN_SAMPLES_PER_LANGUAGE:
            warnings.append(ValidationWarning(
                f"Training data for '{lang}' has only {len(texts)} samples "
                f"(recommend >= {MIN_SAMPLES_PER_LANGUAGE}).",
                severity="warning"
            ))
    
    return warnings


def validate_jaccard(value: float, context: str = "") -> None:
    """Validate that a Jaccard coefficient is in valid range.
    
    Args:
        value: Jaccard coefficient to validate
        context: Description of where this value came from
        
    Raises:
        ValidationError: If value is outside [0, 1]
    """
    if not (0.0 <= value <= 1.0):
        raise ValidationError(
            f"Invalid Jaccard coefficient: {value}. "
            f"Jaccard MUST be in [0, 1]. "
            f"Context: {context}"
        )


def validate_experiment_setup(verbose: bool = True) -> Tuple[bool, List[ValidationWarning]]:
    """Run all validation checks before starting experiments.
    
    Args:
        verbose: If True, print validation results
        
    Returns:
        Tuple of (all_passed: bool, warnings: List[ValidationWarning])
    """
    all_warnings = []
    
    if verbose:
        print("=" * 60)
        print("RESEARCH VALIDATION CHECK")
        print("=" * 60)
    
    # Configuration validation
    if verbose:
        print("\n1. Validating configuration...")
    config_warnings = validate_config()
    all_warnings.extend(config_warnings)
    
    if verbose:
        if config_warnings:
            for w in config_warnings:
                print(f"   {w}")
        else:
            print("   [OK] Configuration OK")
    
    # Check for critical errors
    errors = [w for w in all_warnings if w.severity == "error"]
    warnings_only = [w for w in all_warnings if w.severity == "warning"]
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"VALIDATION SUMMARY: {len(errors)} errors, {len(warnings_only)} warnings")
        print("=" * 60)
        
        if errors:
            print("\n[X] CRITICAL ERRORS (must fix before running experiments):")
            for e in errors:
                print(f"   - {e.message}")

        if warnings_only:
            print("\n[!] WARNINGS (recommended to address):")
            for w in warnings_only:
                print(f"   - {w.message}")

        if not errors and not warnings_only:
            print("\n[OK] All validation checks passed!")
    
    all_passed = len(errors) == 0
    return all_passed, all_warnings


if __name__ == "__main__":
    passed, warnings = validate_experiment_setup(verbose=True)
    sys.exit(0 if passed else 1)
