"""
Result Validation Module

Provides calibration validation, Brier score calculation, and other
metrics for validating Bayesian confidence scores.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""
    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    n_bins: int
    bin_counts: List[int]
    bin_confidences: List[float]
    bin_accuracies: List[float]
    calibration_difference: List[float]  # confidence - accuracy per bin


@dataclass
class CalibrationSummary:
    """High-level calibration summary for reporting."""
    n_samples: int
    positive_rate: float
    mean_confidence: float
    brier_score: float
    ece: float
    mce: float
    n_bins: int


# --------------- helpers ---------------

_EPS = 1e-15  # small epsilon for numerical stability


def _sanitize_array(values: List[float], name: str = "values") -> np.ndarray:
    """Convert list to numpy array and validate no NaN/Inf values."""
    arr = np.array(values, dtype=np.float64).ravel()
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains Inf values")
    return arr


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Numerically stable logit: log(p / (1 - p)), clamping p to (eps, 1-eps)."""
    clamped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(clamped / (1.0 - clamped))


# --------------- validation ---------------


def validate_calibration_inputs(
    predictions: List[float],
    outcomes: List[int]
) -> Tuple[bool, List[str]]:
    """
    Validate basic calibration inputs.

    Args:
        predictions: List of confidence scores (0-1)
        outcomes: List of actual outcomes (0 or 1)

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues: List[str] = []
    if len(predictions) != len(outcomes):
        issues.append("Predictions and outcomes must have same length")

    if len(predictions) == 0:
        issues.append("No samples provided for calibration")

    for i, p in enumerate(predictions):
        if isinstance(p, float) and (math.isnan(p) or math.isinf(p)):
            issues.append(f"Prediction {i} is NaN or Inf: {p}")
        elif p < 0.0 or p > 1.0:
            issues.append(f"Prediction {i} out of bounds: {p}")

    for i, o in enumerate(outcomes):
        if o not in (0, 1):
            issues.append(f"Outcome {i} not binary: {o}")

    return len(issues) == 0, issues


# --------------- metrics ---------------


def brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Calculate Brier Score (Mean Squared Error).
    
    Lower is better: 0 = perfect, 1 = worst.
    
    Args:
        predictions: List of confidence scores (0-1)
        outcomes: List of actual outcomes (0 or 1)
    
    Returns:
        Brier score (float)
    """
    if not predictions or not outcomes:
        return 1.0
    
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    p = _sanitize_array(predictions, "predictions")
    o = _sanitize_array(outcomes, "outcomes")
    
    return float(np.mean((p - o) ** 2))


def expected_calibration_error(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        predictions: List of confidence scores (0-1)
        outcomes: List of actual outcomes (0 or 1)
        n_bins: Number of bins for grouping
    
    Returns:
        Tuple of (ECE, MCE)
    """
    if not predictions or not outcomes:
        return 1.0, 1.0

    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    
    p = _sanitize_array(predictions, "predictions")
    o = _sanitize_array(outcomes, "outcomes")
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total_weight = len(p)
    
    for i in range(n_bins):
        bin_mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        # Include last bin
        if i == n_bins - 1:
            bin_mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
        
        if np.sum(bin_mask) == 0:
            continue
        
        bin_preds = p[bin_mask]
        bin_outcomes = o[bin_mask]
        
        avg_confidence = float(np.mean(bin_preds))
        avg_accuracy = float(np.mean(bin_outcomes))
        weight = np.sum(bin_mask) / total_weight
        
        cal_diff = abs(avg_confidence - avg_accuracy)
        ece += weight * cal_diff
        mce = max(mce, cal_diff)
    
    return ece, mce


def calibrate_confidence(
    predictions: List[float],
    outcomes: List[int],
    method: str = "platt"
) -> Tuple[List[float], Dict]:
    """
    Apply calibration to confidence scores.
    
    Methods:
        - platt: Platt scaling (logistic regression)
        - isotonic: Isotonic regression
        - temperature: Temperature scaling
    
    Args:
        predictions: List of confidence scores (0-1)
        outcomes: List of actual outcomes (0 or 1)
        method: Calibration method
    
    Returns:
        Tuple of (calibrated predictions, metrics dict)
    """
    if len(predictions) < 10:
        # Not enough data for calibration
        return predictions, {"method": "none", "reason": "insufficient_data"}

    valid, issues = validate_calibration_inputs(predictions, outcomes)
    if not valid:
        return predictions, {"method": "none", "reason": "invalid_inputs", "issues": issues}

    # 1-D array for temperature/isotonic, 2-D for Platt (sklearn needs 2-D features)
    p_flat = _sanitize_array(predictions, "predictions")
    o = _sanitize_array(outcomes, "outcomes")
    
    metrics: Dict = {"method": method}
    
    if method == "temperature":
        # Temperature scaling using numerically-stable logit
        from scipy.optimize import minimize_scalar
        
        logits = _safe_logit(p_flat)

        def loss(T):
            scaled = 1.0 / (1.0 + np.exp(-logits / T))
            return float(np.mean((scaled - o) ** 2))
        
        result = minimize_scalar(loss, bounds=(0.1, 10.0), method='bounded')
        T = result.x
        calibrated = 1.0 / (1.0 + np.exp(-logits / T))
        cal_list = calibrated.tolist()
        metrics["temperature"] = float(T)
        metrics["brier_before"] = brier_score(predictions, outcomes)
        metrics["brier_after"] = brier_score(cal_list, outcomes)
        return cal_list, metrics
    
    elif method == "platt":
        # Platt scaling (logistic regression) — needs 2-D features
        from sklearn.linear_model import LogisticRegression
        
        p_2d = p_flat.reshape(-1, 1)
        lr = LogisticRegression(random_state=42)
        lr.fit(p_2d, o)
        calibrated = lr.predict_proba(p_2d)[:, 1]
        cal_list = calibrated.tolist()
        metrics["coefficients"] = {"A": float(lr.coef_[0][0]), "B": float(lr.intercept_[0])}
        metrics["brier_before"] = brier_score(predictions, outcomes)
        metrics["brier_after"] = brier_score(cal_list, outcomes)
        return cal_list, metrics
    
    elif method == "isotonic":
        # Isotonic regression
        from sklearn.isotonic import IsotonicRegression
        
        ir = IsotonicRegression(out_of_bounds="clip")
        calibrated = ir.fit_transform(p_flat, o)
        cal_list = calibrated.tolist()
        metrics["brier_before"] = brier_score(predictions, outcomes)
        metrics["brier_after"] = brier_score(cal_list, outcomes)
        return cal_list, metrics
    
    else:
        return predictions, {"method": "unknown", "error": f"Unknown method: {method}"}


def analyze_calibration(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> CalibrationResult:
    """
    Perform full calibration analysis.
    
    Args:
        predictions: List of confidence scores (0-1)
        outcomes: List of actual outcomes (0 or 1)
        n_bins: Number of bins
    
    Returns:
        CalibrationResult with all metrics
    """
    bs = brier_score(predictions, outcomes)
    ece, mce = expected_calibration_error(predictions, outcomes, n_bins)
    
    p = np.array(predictions, dtype=np.float64)
    o = np.array(outcomes, dtype=np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    bin_counts = []
    bin_confidences = []
    bin_accuracies = []
    calibration_diffs = []
    
    for i in range(n_bins):
        if i < n_bins - 1:
            bin_mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        else:
            bin_mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
        
        if np.sum(bin_mask) == 0:
            bin_counts.append(0)
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)
            calibration_diffs.append(0.0)
            continue
        
        bin_confidences.append(float(np.mean(p[bin_mask])))
        bin_accuracies.append(float(np.mean(o[bin_mask])))
        calibration_diffs.append(bin_confidences[-1] - bin_accuracies[-1])
        bin_counts.append(int(np.sum(bin_mask)))
    
    return CalibrationResult(
        brier_score=bs,
        ece=ece,
        mce=mce,
        n_bins=n_bins,
        bin_counts=bin_counts,
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        calibration_difference=calibration_diffs
    )


def summarize_calibration(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> CalibrationSummary:
    """
    Produce a compact summary for reporting calibration quality.

    Args:
        predictions: List of confidence scores (0-1)
        outcomes: List of actual outcomes (0 or 1)
        n_bins: Number of bins

    Returns:
        CalibrationSummary with aggregate metrics
    """
    if not predictions:
        return CalibrationSummary(
            n_samples=0,
            positive_rate=0.0,
            mean_confidence=0.0,
            brier_score=1.0,
            ece=1.0,
            mce=1.0,
            n_bins=n_bins,
        )

    result = analyze_calibration(predictions, outcomes, n_bins)
    positive_rate = float(np.mean(outcomes)) if outcomes else 0.0
    mean_confidence = float(np.mean(predictions)) if predictions else 0.0
    return CalibrationSummary(
        n_samples=len(predictions),
        positive_rate=positive_rate,
        mean_confidence=mean_confidence,
        brier_score=result.brier_score,
        ece=result.ece,
        mce=result.mce,
        n_bins=n_bins,
    )


def validate_bounds(values: List[float]) -> Tuple[bool, List[str]]:
    """
    Validate that all values are within valid bounds [0, 1] and finite.
    
    Args:
        values: List of confidence scores
    
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues: List[str] = []
    for i, v in enumerate(values):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            issues.append(f"Value {i} is NaN or Inf: {v}")
        elif v < 0.0:
            issues.append(f"Value {i} below 0: {v}")
        elif v > 1.0:
            issues.append(f"Value {i} above 1: {v}")
    
    return len(issues) == 0, issues


def validate_monotonicity(
    timestamps: List[str],
    values: List[float],
    max_jump: float = 0.4
) -> Tuple[bool, List[str]]:
    """
    Check for unexpected value changes.
    
    Args:
        timestamps: ISO8601 timestamps
        values: Confidence scores
        max_jump: Maximum allowed absolute change between points
    
    Returns:
        Tuple of (is_valid, list of anomalies)
    """
    if len(timestamps) != len(values):
        return False, ["timestamps and values must have same length"]

    if len(timestamps) < 2:
        return True, []
    
    if max_jump <= 0.0:
        return False, ["max_jump must be positive"]

    issues: List[str] = []
    for i in range(1, len(values)):
        if math.isnan(values[i]) or math.isnan(values[i - 1]):
            issues.append(f"NaN value at index {i}")
            continue
        change = abs(values[i] - values[i - 1])
        if change >= max_jump:  # Large jump
            issues.append(
                f"Large change at {timestamps[i]}: "
                f"{values[i-1]:.2f} -> {values[i]:.2f}"
            )
    
    return len(issues) == 0, issues
