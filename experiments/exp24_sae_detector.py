"""Experiment 24: SAE-Based Language Detector

Trains a language classifier using SAE activations to validate that
steering actually modifies the underlying language representation.

Research Questions:
  1. Can SAE activations predict language with high accuracy?
  2. Does steering shift the classifier's predictions?
  3. Which SAE features are most predictive of language?

Falsification Criteria:
  - If classifier accuracy is low (<60%) → SAE doesn't capture language
  - If steering doesn't change predictions → steering is superficial
  - If random features work as well → no language-specific features

Methodology:
  1. Extract SAE activations for texts in each language
  2. Train LogisticRegression classifier on mean-pooled activations
  3. Evaluate baseline accuracy
  4. Apply steering and measure prediction shift
  5. Analyze feature importance for language detection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from config import (
    TARGET_LAYERS,
    N_SAMPLES_DISCOVERY,
    N_STEERING_EVAL,
    SEED,
    INDIC_LANGS,
)
from data import load_research_data
from model import GemmaWithSAE
from reproducibility import seed_everything


@dataclass
class ClassifierResult:
    """Results of training a language classifier at a layer."""
    layer: int
    accuracy: float
    cv_accuracy: float
    cv_std: float
    n_train: int
    n_test: int
    languages: List[str]
    per_language_f1: Dict[str, float]
    confusion_matrix: List[List[int]]


@dataclass
class SteeringShiftResult:
    """Results of measuring prediction shift after steering."""
    source_lang: str
    target_lang: str
    layer: int
    baseline_pred_source: float  # % predicted as source before steering
    baseline_pred_target: float  # % predicted as target before steering
    steered_pred_source: float   # % predicted as source after steering
    steered_pred_target: float   # % predicted as target after steering
    prediction_shift: float      # Change in target prediction %


@dataclass
class FeatureImportanceResult:
    """Feature importance analysis results."""
    layer: int
    top_features_per_lang: Dict[str, List[Tuple[int, float]]]  # lang -> [(feat_idx, importance)]
    shared_important_features: List[int]  # Features important for multiple languages
    language_specific_features: Dict[str, List[int]]  # Features unique to each language


def extract_sae_features(
    model: GemmaWithSAE,
    texts: List[str],
    layer: int,
    pooling: str = "mean",
) -> np.ndarray:
    """Extract SAE activation features for a list of texts.

    Args:
        model: Model with SAE
        texts: Input texts
        layer: Layer to extract from
        pooling: How to pool token activations ("mean", "max", "last")

    Returns:
        Array of shape (n_texts, n_features)
    """
    features = []

    for text in texts:
        acts = model.get_sae_activations(text, layer)  # (n_tokens, n_features)

        if acts.shape[0] == 0:
            # Empty activation, skip
            continue

        if pooling == "mean":
            pooled = acts.mean(dim=0)
        elif pooling == "max":
            pooled = acts.max(dim=0).values
        elif pooling == "last":
            pooled = acts[-1]
        else:
            pooled = acts.mean(dim=0)

        features.append(pooled.cpu().numpy())

    return np.array(features)


def train_language_classifier(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    max_samples_per_lang: int = 100,
    test_size: float = 0.2,
) -> Tuple[LogisticRegression, ClassifierResult, LabelEncoder]:
    """Train a language classifier using SAE activations.

    Args:
        model: Model with SAE
        texts_by_lang: Dictionary mapping language codes to texts
        layer: Layer to use for features
        max_samples_per_lang: Maximum samples per language
        test_size: Fraction for test set

    Returns:
        Trained classifier, results, and label encoder
    """
    print(f"  Extracting features from layer {layer}...")

    X_all = []
    y_all = []

    for lang, texts in texts_by_lang.items():
        texts_subset = texts[:max_samples_per_lang]
        features = extract_sae_features(model, texts_subset, layer)

        if len(features) == 0:
            continue

        X_all.append(features)
        y_all.extend([lang] * len(features))

    X = np.vstack(X_all)
    y = np.array(y_all)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=SEED, stratify=y_encoded
    )

    print(f"  Training classifier (train={len(X_train)}, test={len(X_test)})...")

    # Train classifier with regularization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            max_iter=1000,
            random_state=SEED,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
        )
        clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y_encoded, cv=5)
    cv_accuracy = cv_scores.mean()
    cv_std = cv_scores.std()

    # Per-language metrics
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    per_lang_f1 = {lang: report[lang]["f1-score"] for lang in le.classes_}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    result = ClassifierResult(
        layer=layer,
        accuracy=accuracy,
        cv_accuracy=cv_accuracy,
        cv_std=cv_std,
        n_train=len(X_train),
        n_test=len(X_test),
        languages=list(le.classes_),
        per_language_f1=per_lang_f1,
        confusion_matrix=cm,
    )

    return clf, result, le


def compute_steering_vector(
    model: GemmaWithSAE,
    source_texts: List[str],
    target_texts: List[str],
    layer: int,
) -> torch.Tensor:
    """Compute steering vector from source to target language."""
    source_acts = []
    target_acts = []

    for text in source_texts[:50]:
        acts = model.get_sae_activations(text, layer)
        if acts.shape[0] > 0:
            source_acts.append(acts.mean(dim=0))

    for text in target_texts[:50]:
        acts = model.get_sae_activations(text, layer)
        if acts.shape[0] > 0:
            target_acts.append(acts.mean(dim=0))

    if not source_acts or not target_acts:
        return None

    source_mean = torch.stack(source_acts).mean(dim=0)
    target_mean = torch.stack(target_acts).mean(dim=0)

    return target_mean - source_mean


def evaluate_steering_shift(
    model: GemmaWithSAE,
    clf: LogisticRegression,
    le: LabelEncoder,
    source_lang: str,
    target_lang: str,
    source_texts: List[str],
    target_texts: List[str],
    test_texts: List[str],
    layer: int,
    steering_strength: float = 2.0,
) -> SteeringShiftResult:
    """Evaluate how much steering shifts classifier predictions.

    Args:
        model: Model with SAE
        clf: Trained classifier
        le: Label encoder
        source_lang: Source language code
        target_lang: Target language code
        source_texts: Texts for computing steering vector
        target_texts: Texts for computing steering vector
        test_texts: Texts to test on (in source language)
        layer: Layer for steering
        steering_strength: Multiplier for steering vector

    Returns:
        SteeringShiftResult with prediction shifts
    """
    # Compute steering vector
    steering_vec = compute_steering_vector(model, source_texts, target_texts, layer)

    if steering_vec is None:
        return None

    # Get baseline predictions (before steering)
    baseline_texts = test_texts[:30]
    baseline_features = extract_sae_features(model, baseline_texts, layer)
    baseline_probs = clf.predict_proba(baseline_features)

    source_idx = np.where(le.classes_ == source_lang)[0][0]
    target_idx = np.where(le.classes_ == target_lang)[0][0]

    baseline_source_prob = baseline_probs[:, source_idx].mean()
    baseline_target_prob = baseline_probs[:, target_idx].mean()

    # Apply steering via generation, then re-extract SAE features.
    steered_texts = []
    for txt in baseline_texts:
        steered = model.generate_with_steering(
            txt,
            layer=layer,
            steering_vector=steering_vec,
            strength=steering_strength,
            max_new_tokens=0,  # steer prompt itself; classification is on prompt text
            schedule="prompt_only",
        )
        # generate_with_steering returns generated suffix; recombine
        steered_full = txt + steered
        steered_texts.append(steered_full)

    steered_features = extract_sae_features(model, steered_texts, layer)
    steered_probs = clf.predict_proba(steered_features)

    steered_source_prob = steered_probs[:, source_idx].mean()
    steered_target_prob = steered_probs[:, target_idx].mean()

    prediction_shift = steered_target_prob - baseline_target_prob

    return SteeringShiftResult(
        source_lang=source_lang,
        target_lang=target_lang,
        layer=layer,
        baseline_pred_source=baseline_source_prob,
        baseline_pred_target=baseline_target_prob,
        steered_pred_source=steered_source_prob,
        steered_pred_target=steered_target_prob,
        prediction_shift=prediction_shift,
    )


def analyze_feature_importance(
    clf: LogisticRegression,
    le: LabelEncoder,
    layer: int,
    top_k: int = 50,
) -> FeatureImportanceResult:
    """Analyze which SAE features are most important for language classification.

    Args:
        clf: Trained classifier
        le: Label encoder
        layer: Layer number
        top_k: Number of top features to return per language

    Returns:
        FeatureImportanceResult with feature importance analysis
    """
    # Get coefficients (shape: n_classes x n_features)
    coefs = clf.coef_

    top_features_per_lang = {}
    language_specific_features = {}

    for i, lang in enumerate(le.classes_):
        lang_coefs = coefs[i]

        # Get top features by absolute importance
        top_indices = np.argsort(np.abs(lang_coefs))[-top_k:][::-1]
        top_features_per_lang[lang] = [
            (int(idx), float(lang_coefs[idx])) for idx in top_indices
        ]

        # Features where this language has highest coefficient
        lang_specific = []
        for idx in top_indices[:20]:
            # Check if this feature's coefficient is highest for this language
            all_coefs_for_feat = coefs[:, idx]
            if np.argmax(all_coefs_for_feat) == i:
                lang_specific.append(int(idx))

        language_specific_features[lang] = lang_specific

    # Find shared important features (important for multiple languages)
    all_top_features = set()
    for features in top_features_per_lang.values():
        all_top_features.update([f[0] for f in features[:20]])

    shared_features = []
    for feat in all_top_features:
        # Count how many languages this feature is important for
        count = sum(
            1 for lang_feats in top_features_per_lang.values()
            if feat in [f[0] for f in lang_feats[:20]]
        )
        if count >= 2:
            shared_features.append(feat)

    return FeatureImportanceResult(
        layer=layer,
        top_features_per_lang=top_features_per_lang,
        shared_important_features=shared_features,
        language_specific_features=language_specific_features,
    )


def test_random_baseline(
    model: GemmaWithSAE,
    texts_by_lang: Dict[str, List[str]],
    layer: int,
    n_random_features: int = 100,
    n_trials: int = 5,
) -> Tuple[float, float]:
    """Test classifier accuracy with random feature subsets.

    Returns mean accuracy and std across trials.
    """
    sae = model.load_sae(layer)
    n_features = sae.cfg.d_sae

    accuracies = []

    for trial in range(n_trials):
        # Select random features
        np.random.seed(SEED + trial)
        random_features = np.random.choice(n_features, n_random_features, replace=False)

        # Extract only those features
        X_all = []
        y_all = []

        for lang, texts in texts_by_lang.items():
            for text in texts[:50]:
                acts = model.get_sae_activations(text, layer)
                if acts.shape[0] > 0:
                    pooled = acts.mean(dim=0).cpu().numpy()
                    X_all.append(pooled[random_features])
                    y_all.append(lang)

        X = np.array(X_all)
        y = np.array(y_all)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=500, random_state=SEED)
            clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


def main():
    seed_everything(SEED)

    print("=" * 60)
    print("EXPERIMENT 24: SAE-Based Language Detector")
    print("=" * 60)

    # Load model
    model = GemmaWithSAE()
    model.load_model()
    suffix = "_9b" if "9b" in model.model_id.lower() else ""

    # Load data
    data = load_research_data(
        max_train_samples=N_SAMPLES_DISCOVERY,
        max_test_samples=0,
        max_eval_samples=0,
        seed=SEED,
    )

    # Use available languages
    available_langs = [lang for lang in INDIC_LANGS if lang in data.train]
    print(f"\nLanguages: {available_langs}")

    texts_by_lang = {lang: data.train[lang] for lang in available_langs}

    # Test across layers
    classifier_results = []
    steering_results = []
    importance_results = []

    print("\n" + "=" * 60)
    print("TRAINING LANGUAGE CLASSIFIERS")
    print("=" * 60)

    best_layer = None
    best_accuracy = 0
    best_clf = None
    best_le = None

    for layer in TARGET_LAYERS[:5]:  # Test subset of layers
        print(f"\n--- Layer {layer} ---")

        clf, result, le = train_language_classifier(
            model, texts_by_lang, layer,
            max_samples_per_lang=100,
        )

        classifier_results.append(result)

        print(f"  Accuracy: {result.accuracy:.1%}")
        print(f"  CV Accuracy: {result.cv_accuracy:.1%} (+/- {result.cv_std:.1%})")
        print(f"  Per-language F1:")
        for lang, f1 in result.per_language_f1.items():
            print(f"    {lang}: {f1:.2f}")

        if result.accuracy > best_accuracy:
            best_accuracy = result.accuracy
            best_layer = layer
            best_clf = clf
            best_le = le

    print(f"\nBest layer: {best_layer} (accuracy: {best_accuracy:.1%})")

    # Feature importance analysis
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    importance = analyze_feature_importance(best_clf, best_le, best_layer)
    importance_results.append(importance)

    print(f"\nShared important features: {len(importance.shared_important_features)}")
    for lang, features in importance.language_specific_features.items():
        print(f"  {lang}-specific features: {len(features)}")

    # Random baseline comparison
    print("\n" + "=" * 60)
    print("RANDOM BASELINE COMPARISON")
    print("=" * 60)

    random_mean, random_std = test_random_baseline(
        model, texts_by_lang, best_layer, n_random_features=100
    )
    print(f"Random features accuracy: {random_mean:.1%} (+/- {random_std:.1%})")
    print(f"Full features accuracy: {best_accuracy:.1%}")
    print(f"Improvement over random: {best_accuracy - random_mean:.1%}")

    # Steering shift analysis
    print("\n" + "=" * 60)
    print("STEERING SHIFT ANALYSIS")
    print("=" * 60)

    steering_pairs = [
        ("hi", "ta"),  # Indo-Aryan to Dravidian
        ("ta", "hi"),  # Dravidian to Indo-Aryan
        ("hi", "bn"),  # Within Indo-Aryan
    ]

    for source, target in steering_pairs:
        if source not in texts_by_lang or target not in texts_by_lang:
            continue

        print(f"\n  Steering: {source} → {target}")

        result = evaluate_steering_shift(
            model, best_clf, best_le,
            source, target,
            texts_by_lang[source], texts_by_lang[target],
            texts_by_lang[source][:30],
            best_layer,
        )

        if result:
            steering_results.append(result)
            print(f"    Baseline P({source}): {result.baseline_pred_source:.1%}")
            print(f"    Baseline P({target}): {result.baseline_pred_target:.1%}")
            print(f"    Steered P({source}): {result.steered_pred_source:.1%}")
            print(f"    Steered P({target}): {result.steered_pred_target:.1%}")
            print(f"    Prediction shift: {result.prediction_shift:+.1%}")

    # Conclusions
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    # Check if SAE captures language
    sae_captures_language = best_accuracy > 0.6
    print(f"\n  SAE captures language: {'YES' if sae_captures_language else 'NO'} (acc={best_accuracy:.1%})")

    # Check if steering shifts predictions
    avg_shift = np.mean([r.prediction_shift for r in steering_results]) if steering_results else 0
    steering_effective = avg_shift > 0.1
    print(f"  Steering shifts predictions: {'YES' if steering_effective else 'NO'} (avg shift={avg_shift:.1%})")

    # Check if better than random
    better_than_random = best_accuracy > random_mean + 2 * random_std
    print(f"  Better than random: {'YES' if better_than_random else 'NO'}")

    # Overall validation
    validated = sae_captures_language and steering_effective and better_than_random
    print(f"\n  OVERALL: SAE-based detection {'VALIDATED' if validated else 'NOT VALIDATED'}")

    # Save results
    results = {
        "classifier_results": [
            {
                "layer": r.layer,
                "accuracy": r.accuracy,
                "cv_accuracy": r.cv_accuracy,
                "cv_std": r.cv_std,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "languages": r.languages,
                "per_language_f1": r.per_language_f1,
            }
            for r in classifier_results
        ],
        "best_layer": best_layer,
        "best_accuracy": best_accuracy,
        "random_baseline": {
            "mean": random_mean,
            "std": random_std,
        },
        "steering_results": [
            {
                "source_lang": r.source_lang,
                "target_lang": r.target_lang,
                "layer": r.layer,
                "baseline_pred_source": r.baseline_pred_source,
                "baseline_pred_target": r.baseline_pred_target,
                "steered_pred_source": r.steered_pred_source,
                "steered_pred_target": r.steered_pred_target,
                "prediction_shift": r.prediction_shift,
            }
            for r in steering_results
        ],
        "feature_importance": {
            "layer": importance.layer,
            "n_shared_features": len(importance.shared_important_features),
            "language_specific_counts": {
                lang: len(feats)
                for lang, feats in importance.language_specific_features.items()
            },
        },
        "validation": {
            "sae_captures_language": sae_captures_language,
            "steering_shifts_predictions": steering_effective,
            "better_than_random": better_than_random,
            "overall_validated": validated,
        },
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp24_sae_detector{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
