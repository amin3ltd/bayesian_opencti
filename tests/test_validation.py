"""
Tests for validation and calibration module.
"""

import pytest

pytestmark = pytest.mark.unit
import numpy as np
from validation.calibration import (
    brier_score,
    expected_calibration_error,
    analyze_calibration,
    calibrate_confidence,
    summarize_calibration,
    validate_calibration_inputs,
    validate_bounds,
    validate_monotonicity,
)


class TestBrierScore:
    """Tests for Brier Score calculation."""

    def test_perfect_prediction(self):
        """Test Brier score of 0 for perfect predictions."""
        preds = [1.0, 0.0, 1.0, 0.0, 1.0]
        outcomes = [1, 0, 1, 0, 1]
        score = brier_score(preds, outcomes)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_worst_prediction(self):
        """Test Brier score of 1 for worst predictions."""
        preds = [0.0, 1.0, 0.0, 1.0, 0.0]
        outcomes = [1, 0, 1, 0, 1]
        score = brier_score(preds, outcomes)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_random_prediction(self):
        """Test Brier score for random predictions."""
        np.random.seed(42)
        preds = np.random.uniform(0, 1, 100).tolist()
        outcomes = np.random.randint(0, 2, 100).tolist()
        score = brier_score(preds, outcomes)
        
        assert 0.0 < score < 1.0

    def test_empty_input(self):
        """Test Brier score returns 1.0 for empty input."""
        score = brier_score([], [])
        assert score == 1.0

    def test_mismatched_lengths(self):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError):
            brier_score([0.5, 0.6], [0, 1, 1])


class TestExpectedCalibrationError:
    """Tests for ECE calculation."""

    def test_perfect_calibration(self):
        """Test ECE of 0 for perfectly calibrated predictions."""
        # Create perfectly calibrated: all 0.8s with 80% accuracy
        preds = [0.8] * 100
        outcomes = [1 if i < 80 else 0 for i in range(100)]
        ece, mce = expected_calibration_error(preds, outcomes, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-3)

    def test_miscalibrated_predictions(self):
        """Test ECE for miscalibrated predictions."""
        # High confidence but low accuracy
        preds = [0.9] * 100
        outcomes = [0] * 100  # All wrong
        ece, mce = expected_calibration_error(preds, outcomes, n_bins=10)
        assert ece > 0.1  # Should be high

    def test_empty_input(self):
        """Test ECE returns 1.0 for empty input."""
        ece, mce = expected_calibration_error([], [], n_bins=10)
        assert ece == 1.0
        assert mce == 1.0


class TestAnalyzeCalibration:
    """Tests for full calibration analysis."""

    def test_analysis_output_structure(self):
        """Test that analysis returns all required fields."""
        preds = [0.8, 0.9, 0.7, 0.6, 0.5]
        outcomes = [1, 1, 1, 0, 0]
        result = analyze_calibration(preds, outcomes, n_bins=5)
        
        assert hasattr(result, 'brier_score')
        assert hasattr(result, 'ece')
        assert hasattr(result, 'mce')
        assert hasattr(result, 'bin_counts')
        assert hasattr(result, 'bin_confidences')
        assert hasattr(result, 'bin_accuracies')
        assert hasattr(result, 'calibration_difference')

    def test_bin_counts_sum(self):
        """Test that bin counts sum to total predictions."""
        np.random.seed(42)
        preds = np.random.uniform(0, 1, 1000).tolist()
        outcomes = np.random.randint(0, 2, 1000).tolist()
        result = analyze_calibration(preds, outcomes, n_bins=10)
        
        assert sum(result.bin_counts) == 1000


class TestCalibrateConfidence:
    """Tests for confidence calibration methods."""

    def test_temperature_scaling(self):
        """Test temperature scaling calibration."""
        np.random.seed(42)
        preds = np.random.uniform(0, 1, 100).tolist()
        outcomes = np.random.randint(0, 2, 100).tolist()
        
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="temperature")
        
        assert len(calibrated) == len(preds)
        assert "temperature" in metrics
        assert "brier_before" in metrics
        assert "brier_after" in metrics

    def test_insufficient_data(self):
        """Test calibration with insufficient data returns original."""
        preds = [0.5, 0.6, 0.7]
        outcomes = [1, 1, 0]
        
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="platt")
        
        assert calibrated == preds
        assert metrics["method"] == "none"


class TestCalibrationInputValidation:
    """Tests for calibration input validation."""

    def test_valid_inputs(self):
        """Test validation for correct inputs."""
        preds = [0.1, 0.5, 0.9]
        outcomes = [0, 1, 1]
        is_valid, issues = validate_calibration_inputs(preds, outcomes)
        assert is_valid is True
        assert issues == []

    def test_length_mismatch(self):
        """Test detection of length mismatch."""
        preds = [0.1, 0.2]
        outcomes = [1]
        is_valid, issues = validate_calibration_inputs(preds, outcomes)
        assert is_valid is False
        assert any("same length" in issue for issue in issues)

    def test_out_of_bounds_prediction(self):
        """Test detection of out-of-range predictions."""
        preds = [1.2]
        outcomes = [1]
        is_valid, issues = validate_calibration_inputs(preds, outcomes)
        assert is_valid is False
        assert any("out of bounds" in issue for issue in issues)

    def test_non_binary_outcome(self):
        """Test detection of non-binary outcomes."""
        preds = [0.2, 0.3]
        outcomes = [0, 2]
        is_valid, issues = validate_calibration_inputs(preds, outcomes)
        assert is_valid is False
        assert any("not binary" in issue for issue in issues)

    def test_empty_inputs(self):
        """Test detection of empty inputs."""
        is_valid, issues = validate_calibration_inputs([], [])
        assert is_valid is False
        assert any("No samples" in issue for issue in issues)


class TestCalibrationSummary:
    """Tests for calibration summary output."""

    def test_summary_values(self):
        """Test summary aggregates."""
        preds = [0.2, 0.8, 0.6, 0.4]
        outcomes = [0, 1, 1, 0]
        summary = summarize_calibration(preds, outcomes, n_bins=4)

        assert summary.n_samples == 4
        assert summary.positive_rate == pytest.approx(0.5)
        assert summary.mean_confidence == pytest.approx(0.5)
        assert summary.n_bins == 4

    def test_summary_empty(self):
        """Test summary for empty inputs."""
        summary = summarize_calibration([], [], n_bins=5)
        assert summary.n_samples == 0
        assert summary.brier_score == pytest.approx(1.0)


class TestValidateBounds:
    """Tests for bounds validation."""

    def test_valid_values(self):
        """Test validation of valid values."""
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        is_valid, issues = validate_bounds(values)
        assert is_valid is True
        assert len(issues) == 0

    def test_invalid_negative(self):
        """Test detection of negative values."""
        values = [0.5, -0.1, 0.8]
        is_valid, issues = validate_bounds(values)
        assert is_valid is False
        assert len(issues) == 1

    def test_invalid_above_one(self):
        """Test detection of values above 1."""
        values = [0.5, 1.5, 0.8]
        is_valid, issues = validate_bounds(values)
        assert is_valid is False
        assert len(issues) == 1


class TestValidateMonotonicity:
    """Tests for monotonicity validation."""

    def test_normal_changes(self):
        """Test validation of normal gradual changes."""
        timestamps = ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]
        values = [0.5, 0.52, 0.48]
        is_valid, issues = validate_monotonicity(timestamps, values)
        assert is_valid is True

    def test_large_jump_detected(self):
        """Test detection of large value jumps."""
        timestamps = ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]
        values = [0.5, 0.95]  # Jump of 0.45
        is_valid, issues = validate_monotonicity(timestamps, values)
        assert is_valid is False
        assert len(issues) == 1

    def test_single_value(self):
        """Test with single value returns valid."""
        timestamps = ["2024-01-01T00:00:00Z"]
        values = [0.5]
        is_valid, issues = validate_monotonicity(timestamps, values)
        assert is_valid is True
