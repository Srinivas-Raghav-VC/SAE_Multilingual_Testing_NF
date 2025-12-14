"""Statistical Testing Framework for SAE Multilingual Steering Research.

This module provides rigorous statistical testing utilities:
- Bootstrap confidence intervals
- Paired significance tests (t-test, Wilcoxon)
- Effect size calculations (Cohen's d, rank-biserial)
- Multiple comparison correction (Bonferroni, Holm-Bonferroni, FDR)
- Per-prompt variance reporting

References:
- Efron & Tibshirani (1993) for bootstrap methods
- Cohen (1988) for effect sizes
- Benjamini & Hochberg (1995) for FDR correction
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_low: float
    ci_high: float
    n1: int
    n2: int
    significant_at_05: bool
    significant_at_01: bool
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            "test": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "ci_95": [self.ci_low, self.ci_high],
            "n": [self.n1, self.n2],
            "significant_05": self.significant_at_05,
            "significant_01": self.significant_at_01,
            "interpretation": self.interpretation,
        }

    def __str__(self) -> str:
        sig = "***" if self.significant_at_01 else ("**" if self.significant_at_05 else "n.s.")
        return (
            f"{self.test_name}: stat={self.statistic:.3f}, p={self.p_value:.4f} {sig}, "
            f"{self.effect_size_name}={self.effect_size:.3f}, "
            f"95% CI=[{self.ci_low:.3f}, {self.ci_high:.3f}]"
        )


@dataclass
class BootstrapResult:
    """Container for bootstrap confidence interval results."""
    estimate: float
    ci_low: float
    ci_high: float
    std_error: float
    n_samples: int
    n_bootstrap: int
    confidence_level: float

    def to_dict(self) -> Dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std_error": self.std_error,
            "n_samples": self.n_samples,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
        }

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] (SE={self.std_error:.3f})"


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for a statistic.

    Uses the percentile method (bias-corrected and accelerated BCa is more
    accurate but computationally expensive; percentile is standard for most uses).

    Args:
        data: 1D array of observations
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        BootstrapResult with estimate, CI, and standard error
    """
    data = np.asarray(data)
    n = len(data)

    if n == 0:
        return BootstrapResult(
            estimate=np.nan, ci_low=np.nan, ci_high=np.nan,
            std_error=np.nan, n_samples=0, n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )

    if n == 1:
        val = float(data[0])
        warnings.warn(
            "bootstrap_ci called with n==1; returning a degenerate CI (ci_low==ci_high) "
            "which should not be interpreted as statistical certainty.",
            RuntimeWarning,
        )
        return BootstrapResult(
            estimate=val, ci_low=val, ci_high=val,
            std_error=0.0, n_samples=1, n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )

    rng = np.random.RandomState(seed)

    # Point estimate
    point_estimate = statistic(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(resample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile CI
    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    # Standard error
    std_error = np.std(bootstrap_stats, ddof=1)

    return BootstrapResult(
        estimate=float(point_estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        std_error=float(std_error),
        n_samples=n,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def bootstrap_difference_ci(
    data1: Union[List[float], np.ndarray],
    data2: Union[List[float], np.ndarray],
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute bootstrap CI for difference between two groups.

    Useful for comparing success rates between methods.
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # Handle empty inputs gracefully
    if len(data1) == 0 or len(data2) == 0:
        return BootstrapResult(
            estimate=float('nan'),
            ci_low=float('nan'),
            ci_high=float('nan'),
            std_error=float('nan'),
            n_samples=len(data1) + len(data2),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )

    rng = np.random.RandomState(seed)

    point_diff = statistic(data1) - statistic(data2)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        resample1 = rng.choice(data1, size=len(data1), replace=True)
        resample2 = rng.choice(data2, size=len(data2), replace=True)
        bootstrap_diffs.append(statistic(resample1) - statistic(resample2))

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return BootstrapResult(
        estimate=float(point_diff),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        std_error=float(np.std(bootstrap_diffs, ddof=1)),
        n_samples=len(data1) + len(data2),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size for two independent groups.

    Uses pooled standard deviation.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_paired(diff: np.ndarray) -> float:
    """Compute Cohen's d for paired samples (using difference scores)."""
    if len(diff) < 2:
        return np.nan

    std = np.std(diff, ddof=1)
    if std == 0:
        return 0.0

    return np.mean(diff) / std


def rank_biserial_correlation(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute rank-biserial correlation (effect size for Mann-Whitney U).

    r = 1 - (2U)/(n1*n2)

    Interpretation similar to Cohen's d thresholds.
    """
    n1, n2 = len(group1), len(group2)

    if n1 == 0 or n2 == 0:
        return np.nan

    try:
        u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        r = 1 - (2 * u_stat) / (n1 * n2)
        return r
    except Exception:
        return np.nan


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d or similar effect size."""
    if np.isnan(d):
        return "undefined"

    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# SIGNIFICANCE TESTS
# =============================================================================

def paired_ttest(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    alternative: str = 'two-sided',
) -> StatisticalResult:
    """Paired t-test for comparing two related samples.

    Use when comparing same prompts across two methods.

    Args:
        group1, group2: Paired observations (same length)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        StatisticalResult with test statistics
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) != len(group2):
        raise ValueError("Paired t-test requires equal-length arrays")

    n = len(group1)
    diff = group1 - group2

    # Handle edge cases
    if n < 2:
        return StatisticalResult(
            test_name="paired_ttest",
            statistic=np.nan,
            p_value=1.0,
            effect_size=np.nan,
            effect_size_name="Cohen's d",
            ci_low=np.nan,
            ci_high=np.nan,
            n1=n,
            n2=n,
            significant_at_05=False,
            significant_at_01=False,
            interpretation="insufficient data"
        )

    t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)

    # Effect size (Cohen's d for paired)
    d = cohens_d_paired(diff)

    # CI for mean difference
    ci = bootstrap_ci(diff, np.mean)

    return StatisticalResult(
        test_name="paired_ttest",
        statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=float(d),
        effect_size_name="Cohen's d",
        ci_low=ci.ci_low,
        ci_high=ci.ci_high,
        n1=n,
        n2=n,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        interpretation=f"mean diff = {np.mean(diff):.3f}, {interpret_effect_size(d)} effect"
    )


def wilcoxon_test(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    alternative: str = 'two-sided',
) -> StatisticalResult:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. More robust when
    normality assumption is violated.
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) != len(group2):
        raise ValueError("Wilcoxon test requires equal-length arrays")

    n = len(group1)
    diff = group1 - group2

    if n < 2:
        return StatisticalResult(
            test_name="wilcoxon",
            statistic=np.nan,
            p_value=1.0,
            effect_size=np.nan,
            effect_size_name="rank-biserial r",
            ci_low=np.nan,
            ci_high=np.nan,
            n1=n,
            n2=n,
            significant_at_05=False,
            significant_at_01=False,
            interpretation="insufficient data"
        )

    # Filter out zeros (ties at 0)
    nonzero_diff = diff[diff != 0]

    if len(nonzero_diff) < 2:
        return StatisticalResult(
            test_name="wilcoxon",
            statistic=np.nan,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="rank-biserial r",
            ci_low=np.nan,
            ci_high=np.nan,
            n1=n,
            n2=n,
            significant_at_05=False,
            significant_at_01=False,
            interpretation="no non-zero differences"
        )

    try:
        stat, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
    except Exception as e:
        warnings.warn(f"Wilcoxon test failed: {e}")
        return StatisticalResult(
            test_name="wilcoxon",
            statistic=np.nan,
            p_value=1.0,
            effect_size=np.nan,
            effect_size_name="rank-biserial r",
            ci_low=np.nan,
            ci_high=np.nan,
            n1=n,
            n2=n,
            significant_at_05=False,
            significant_at_01=False,
            interpretation=f"test failed: {e}"
        )

    # Effect size: signed rank-biserial correlation for Wilcoxon signed-rank.
    #
    # Compute W+ and W- directly from ranks of |diff| (excluding zeros), then:
    #   r = (W+ - W-) / (n(n+1)/2)
    #
    # This yields r in [-1, 1] where positive means group1 > group2.
    n_nonzero = len(nonzero_diff)
    ranks = stats.rankdata(np.abs(nonzero_diff))
    w_pos = float(ranks[nonzero_diff > 0].sum())
    w_neg = float(ranks[nonzero_diff < 0].sum())
    denom = float(n_nonzero * (n_nonzero + 1) / 2.0)
    r = 0.0 if denom == 0 else float((w_pos - w_neg) / denom)

    ci = bootstrap_ci(diff, np.median)

    return StatisticalResult(
        test_name="wilcoxon",
        statistic=float(stat),
        p_value=float(p_value),
        effect_size=float(r),
        effect_size_name="rank-biserial r (signed)",
        ci_low=ci.ci_low,
        ci_high=ci.ci_high,
        n1=n,
        n2=n,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        interpretation=f"median diff = {np.median(diff):.3f}, {interpret_effect_size(r)} effect"
    )


def independent_ttest(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    alternative: str = 'two-sided',
) -> StatisticalResult:
    """Independent samples t-test (Welch's t-test).

    Use when comparing two independent groups (different prompts).
    Uses Welch's correction for unequal variances.
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return StatisticalResult(
            test_name="welch_ttest",
            statistic=np.nan,
            p_value=1.0,
            effect_size=np.nan,
            effect_size_name="Cohen's d",
            ci_low=np.nan,
            ci_high=np.nan,
            n1=n1,
            n2=n2,
            significant_at_05=False,
            significant_at_01=False,
            interpretation="insufficient data"
        )

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False, alternative=alternative)

    d = cohens_d(group1, group2)
    ci = bootstrap_difference_ci(group1, group2, np.mean)

    return StatisticalResult(
        test_name="welch_ttest",
        statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=float(d),
        effect_size_name="Cohen's d",
        ci_low=ci.ci_low,
        ci_high=ci.ci_high,
        n1=n1,
        n2=n2,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        interpretation=f"mean diff = {np.mean(group1) - np.mean(group2):.3f}, {interpret_effect_size(d)} effect"
    )


def mann_whitney_test(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    alternative: str = 'two-sided',
) -> StatisticalResult:
    """Mann-Whitney U test for two independent samples.

    Non-parametric alternative to independent t-test.
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1, n2 = len(group1), len(group2)

    if n1 < 1 or n2 < 1:
        return StatisticalResult(
            test_name="mann_whitney",
            statistic=np.nan,
            p_value=1.0,
            effect_size=np.nan,
            effect_size_name="rank-biserial r",
            ci_low=np.nan,
            ci_high=np.nan,
            n1=n1,
            n2=n2,
            significant_at_05=False,
            significant_at_01=False,
            interpretation="insufficient data"
        )

    try:
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
    except Exception as e:
        return StatisticalResult(
            test_name="mann_whitney",
            statistic=np.nan,
            p_value=1.0,
            effect_size=np.nan,
            effect_size_name="rank-biserial r",
            ci_low=np.nan,
            ci_high=np.nan,
            n1=n1,
            n2=n2,
            significant_at_05=False,
            significant_at_01=False,
            interpretation=f"test failed: {e}"
        )

    r = rank_biserial_correlation(group1, group2)
    ci = bootstrap_difference_ci(group1, group2, np.median)

    return StatisticalResult(
        test_name="mann_whitney",
        statistic=float(u_stat),
        p_value=float(p_value),
        effect_size=float(r),
        effect_size_name="rank-biserial r",
        ci_low=ci.ci_low,
        ci_high=ci.ci_high,
        n1=n1,
        n2=n2,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        interpretation=f"median diff = {np.median(group1) - np.median(group2):.3f}, {interpret_effect_size(r)} effect"
    )


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Bonferroni correction for multiple comparisons.

    Most conservative correction. Use when comparisons are independent
    and Type I error control is critical.
    """
    n = len(p_values)

    # Handle empty input
    if n == 0:
        return {
            "method": "bonferroni",
            "original_p_values": [],
            "adjusted_p_values": [],
            "adjusted_alpha": alpha,
            "significant": [],
            "n_comparisons": 0,
        }

    adjusted_alpha = alpha / n

    adjusted_p = [min(p * n, 1.0) for p in p_values]
    significant = [p < adjusted_alpha for p in p_values]

    return {
        "method": "bonferroni",
        "original_p_values": p_values,
        "adjusted_p_values": adjusted_p,
        "adjusted_alpha": adjusted_alpha,
        "significant": significant,
        "n_comparisons": n,
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni while still controlling FWER.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted_p = np.zeros(n)
    significant = [False] * n

    # Calculate raw adjusted p-values (step-down)
    raw_adjusted = np.zeros(n)
    for i in range(n):
        raw_adjusted[i] = min(sorted_p[i] * (n - i), 1.0)

    # Enforce monotonicity: p_(i) must be <= p_(i+1)
    # Use np.maximum.accumulate for efficient cumulative max
    monotonic_adjusted = np.maximum.accumulate(raw_adjusted)

    # Map back to original indices
    for i, idx in enumerate(sorted_indices):
        adjusted_p[idx] = monotonic_adjusted[i]

    for i, p in enumerate(adjusted_p):
        significant[i] = p < alpha

    return {
        "method": "holm_bonferroni",
        "original_p_values": p_values,
        "adjusted_p_values": adjusted_p.tolist(),
        "adjusted_alpha": alpha,
        "significant": significant,
        "n_comparisons": n,
    }


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Benjamini-Hochberg FDR correction.

    Controls False Discovery Rate instead of FWER. Less conservative,
    more appropriate for exploratory analyses with many comparisons.
    """
    n = len(p_values)

    # Guard against empty input to avoid division-by-zero in critical value computation
    if n == 0:
        return {
            "method": "benjamini_hochberg",
            "original_p_values": [],
            "adjusted_p_values": [],
            "fdr_level": alpha,
            "significant": [],
            "n_comparisons": 0,
            "n_discoveries": 0,
        }
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # BH critical values
    critical_values = [(i + 1) / n * alpha for i in range(n)]

    # Find largest k where p[k] <= (k/n) * alpha
    significant = [False] * n
    max_k = -1

    for k in range(n):
        if sorted_p[k] <= critical_values[k]:
            max_k = k

    if max_k >= 0:
        for i in range(max_k + 1):
            significant[sorted_indices[i]] = True

    # Adjusted p-values (BH)
    adjusted_p = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted_p[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted_p[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted_p[sorted_indices[i + 1]]
            )

    adjusted_p = np.clip(adjusted_p, 0, 1)

    return {
        "method": "benjamini_hochberg",
        "original_p_values": p_values,
        "adjusted_p_values": adjusted_p.tolist(),
        "fdr_level": alpha,
        "significant": significant,
        "n_comparisons": n,
        "n_discoveries": sum(significant),
    }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR STEERING EXPERIMENTS
# =============================================================================

def compare_methods_paired(
    results_by_method: Dict[str, List[float]],
    baseline_method: str = "random",
    test_type: str = "wilcoxon",
) -> Dict[str, StatisticalResult]:
    """Compare multiple methods against a baseline using paired tests.

    Args:
        results_by_method: {method_name: [per_prompt_scores]}
        baseline_method: Method to compare against
        test_type: "wilcoxon" or "ttest"

    Returns:
        Dict of method -> StatisticalResult
    """
    if baseline_method not in results_by_method:
        raise ValueError(f"Baseline method '{baseline_method}' not in results")

    baseline = np.asarray(results_by_method[baseline_method])
    test_fn = wilcoxon_test if test_type == "wilcoxon" else paired_ttest

    comparisons = {}
    for method, scores in results_by_method.items():
        if method == baseline_method:
            continue

        scores = np.asarray(scores)
        if len(scores) != len(baseline):
            warnings.warn(f"Method {method} has different length than baseline, using independent test")
            comparisons[method] = mann_whitney_test(scores, baseline)
        else:
            comparisons[method] = test_fn(scores, baseline)

    return comparisons


def report_with_ci(
    values: Union[List[float], np.ndarray],
    name: str = "metric",
    as_percentage: bool = False,
) -> str:
    """Format a metric with bootstrap CI for reporting.

    Args:
        values: List of per-sample values
        name: Name of the metric
        as_percentage: If True, format as percentage

    Returns:
        Formatted string like "success_rate: 75.2% [68.1%, 82.3%]"
    """
    ci = bootstrap_ci(values)

    if as_percentage:
        return f"{name}: {ci.estimate*100:.1f}% [{ci.ci_low*100:.1f}%, {ci.ci_high*100:.1f}%]"
    else:
        return f"{name}: {ci.estimate:.3f} [{ci.ci_low:.3f}, {ci.ci_high:.3f}]"


def compute_all_pairwise_comparisons(
    results_by_method: Dict[str, List[float]],
    test_type: str = "wilcoxon",
    correction: str = "holm",
) -> Dict:
    """Compute all pairwise comparisons with multiple testing correction.

    Returns comprehensive comparison results for methods.
    """
    methods = list(results_by_method.keys())
    n_methods = len(methods)

    if n_methods < 2:
        return {"error": "Need at least 2 methods to compare"}

    test_fn = wilcoxon_test if test_type == "wilcoxon" else paired_ttest

    # All pairwise comparisons
    comparisons = []
    p_values = []

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            m1, m2 = methods[i], methods[j]
            s1 = np.asarray(results_by_method[m1])
            s2 = np.asarray(results_by_method[m2])

            if len(s1) == len(s2):
                result = test_fn(s1, s2)
            else:
                result = mann_whitney_test(s1, s2)

            comparisons.append({
                "method1": m1,
                "method2": m2,
                "result": result,
            })
            p_values.append(result.p_value)

    # Apply correction
    if correction == "bonferroni":
        corrected = bonferroni_correction(p_values)
    elif correction == "holm":
        corrected = holm_bonferroni_correction(p_values)
    elif correction == "fdr":
        corrected = benjamini_hochberg_correction(p_values)
    else:
        corrected = {"adjusted_p_values": p_values, "significant": [p < 0.05 for p in p_values]}

    # Combine results
    for i, comp in enumerate(comparisons):
        comp["adjusted_p"] = corrected["adjusted_p_values"][i]
        comp["significant_corrected"] = corrected["significant"][i]

    return {
        "comparisons": comparisons,
        "correction_method": correction,
        "n_comparisons": len(comparisons),
        "summary": {
            "n_significant_uncorrected": sum(1 for c in comparisons if c["result"].significant_at_05),
            "n_significant_corrected": sum(1 for c in comparisons if c["significant_corrected"]),
        }
    }


# =============================================================================
# HYPOTHESIS TESTING HELPERS
# =============================================================================

@dataclass
class HypothesisTestResult:
    """Result of testing a specific research hypothesis."""
    hypothesis_id: str
    description: str
    passed: bool
    test_result: Optional[StatisticalResult]
    effect_estimate: float
    threshold: float
    confidence_interval: Tuple[float, float]
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            "hypothesis": self.hypothesis_id,
            "description": self.description,
            "passed": self.passed,
            "effect_estimate": self.effect_estimate,
            "threshold": self.threshold,
            "ci_95": list(self.confidence_interval),
            "interpretation": self.interpretation,
            "test": self.test_result.to_dict() if self.test_result else None,
        }


def test_superiority_hypothesis(
    treatment: Union[List[float], np.ndarray],
    control: Union[List[float], np.ndarray],
    hypothesis_id: str,
    description: str,
    margin: float = 0.0,
    paired: bool = True,
) -> HypothesisTestResult:
    """Test if treatment is superior to control by a margin.

    H0: treatment - control <= margin
    H1: treatment - control > margin

    Args:
        treatment: Treatment group scores
        control: Control group scores
        hypothesis_id: ID like "H2"
        description: Description of hypothesis
        margin: Required superiority margin (e.g., 0.05 for 5%)
        paired: Whether samples are paired
    """
    treatment = np.asarray(treatment)
    control = np.asarray(control)

    if paired and len(treatment) == len(control):
        diff = treatment - control
        effect = np.mean(diff)
        ci = bootstrap_ci(diff)

        # One-sided test; enforce minimum sample size to avoid invalid p-values
        if len(diff) < 2:
            test_result = StatisticalResult(
                test_name="paired_ttest",
                statistic=np.nan,
                p_value=1.0,
                effect_size=np.nan,
                effect_size_name="Cohen's d",
                ci_low=np.nan,
                ci_high=np.nan,
                n1=len(diff),
                n2=len(diff),
                significant_at_05=False,
                significant_at_01=False,
                interpretation="insufficient data",
            )
        else:
            test_result = paired_ttest(treatment, control, alternative='greater')
    else:
        effect = np.mean(treatment) - np.mean(control)
        ci = bootstrap_difference_ci(treatment, control)
        test_result = independent_ttest(treatment, control, alternative='greater')

    # Hypothesis passes if:
    # 1. Effect > margin
    # 2. CI lower bound > 0 (or > margin for stricter test)
    # 3. p < 0.05
    passed = (
        effect > margin and
        ci.ci_low > margin and
        test_result.significant_at_05
    )

    return HypothesisTestResult(
        hypothesis_id=hypothesis_id,
        description=description,
        passed=passed,
        test_result=test_result,
        effect_estimate=effect,
        threshold=margin,
        confidence_interval=(ci.ci_low, ci.ci_high),
        interpretation=f"Effect = {effect:.3f} (threshold: {margin}), 95% CI [{ci.ci_low:.3f}, {ci.ci_high:.3f}]"
    )


def test_threshold_hypothesis(
    values: Union[List[float], np.ndarray],
    hypothesis_id: str,
    description: str,
    threshold: float,
    direction: str = "greater",
) -> HypothesisTestResult:
    """Test if metric exceeds (or is below) a threshold.

    Args:
        values: Observed values
        hypothesis_id: ID like "H1"
        description: Description
        threshold: Threshold value
        direction: "greater" or "less"
    """
    values = np.asarray(values)
    ci = bootstrap_ci(values)

    if direction == "greater":
        # One-sample t-test against threshold (one-sided)
        t_stat, p_value = stats.ttest_1samp(values, threshold, alternative='greater')
        passed = ci.estimate > threshold and ci.ci_low > threshold * 0.9  # Allow 10% slack
    else:
        t_stat, p_value = stats.ttest_1samp(values, threshold, alternative='less')
        passed = ci.estimate < threshold and ci.ci_high < threshold * 1.1

    return HypothesisTestResult(
        hypothesis_id=hypothesis_id,
        description=description,
        passed=passed,
        test_result=None,
        effect_estimate=ci.estimate,
        threshold=threshold,
        confidence_interval=(ci.ci_low, ci.ci_high),
        interpretation=f"Estimate = {ci.estimate:.3f} (threshold: {threshold}), 95% CI [{ci.ci_low:.3f}, {ci.ci_high:.3f}]"
    )


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Testing statistical framework...\n")

    # Test bootstrap CI
    print("1. Bootstrap CI...")
    data = [0.7, 0.8, 0.75, 0.82, 0.78, 0.71, 0.79, 0.83, 0.76, 0.80]
    ci = bootstrap_ci(data)
    print(f"   Mean: {ci}")
    assert 0.7 < ci.estimate < 0.85
    print("   ✓ Bootstrap CI works\n")

    # Test paired t-test
    print("2. Paired t-test...")
    method1 = [0.7, 0.8, 0.75, 0.82, 0.78]
    method2 = [0.6, 0.65, 0.62, 0.68, 0.64]
    result = paired_ttest(method1, method2)
    print(f"   {result}")
    assert result.significant_at_05
    print("   ✓ Paired t-test works\n")

    # Test Wilcoxon
    print("3. Wilcoxon test...")
    result = wilcoxon_test(method1, method2)
    print(f"   {result}")
    print("   ✓ Wilcoxon test works\n")

    # Test multiple comparison correction
    print("4. Multiple comparison correction...")
    p_values = [0.01, 0.04, 0.03, 0.08, 0.002]
    corrected = holm_bonferroni_correction(p_values)
    print(f"   Original: {p_values}")
    print(f"   Adjusted: {corrected['adjusted_p_values']}")
    print(f"   Significant: {corrected['significant']}")
    print("   ✓ Correction works\n")

    # Test hypothesis testing
    print("5. Hypothesis testing...")
    treatment = [0.75, 0.80, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83]
    control = [0.65, 0.68, 0.62, 0.70, 0.66, 0.69, 0.64, 0.67]
    result = test_superiority_hypothesis(
        treatment, control,
        "H2", "Treatment > Control by 5%",
        margin=0.05
    )
    print(f"   {result.interpretation}")
    print(f"   Passed: {result.passed}")
    print("   ✓ Hypothesis testing works\n")

    print("✓ All statistical tests passed!")
