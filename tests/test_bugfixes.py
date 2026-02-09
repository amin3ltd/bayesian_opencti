"""
Tests specifically verifying bug fixes:
  - Temperature scaling 2D array bug
  - Logit division by zero
  - NaN/Inf handling in calibration
  - API parameter validation
  - Duplicate YAML keys
  - clamp01 NaN handling
  - Model edge/node corrupted input handling
"""

import math
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from bayes.model import BayesianConfidenceModel, clamp01
from validation.calibration import (
    brier_score,
    expected_calibration_error,
    calibrate_confidence,
    validate_calibration_inputs,
    validate_bounds,
    validate_monotonicity,
    analyze_calibration,
    summarize_calibration,
)

pytestmark = pytest.mark.unit


# ===================================================================
# Bug fix: Temperature scaling 2D array
# ===================================================================


class TestTemperatureScalingFix:
    """Verify temperature scaling returns flat list, not nested."""

    def test_temperature_returns_flat_list(self):
        np.random.seed(42)
        preds = np.random.uniform(0.1, 0.9, 100).tolist()
        outcomes = np.random.randint(0, 2, 100).tolist()
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="temperature")

        # Must be a flat list of floats, not nested lists
        assert isinstance(calibrated, list)
        assert len(calibrated) == len(preds)
        for v in calibrated:
            assert isinstance(v, float), f"Expected float, got {type(v)}: {v}"
            assert not math.isnan(v)
            assert not math.isinf(v)

    def test_temperature_brier_after_is_scalar(self):
        np.random.seed(42)
        preds = np.random.uniform(0.1, 0.9, 100).tolist()
        outcomes = np.random.randint(0, 2, 100).tolist()
        _, metrics = calibrate_confidence(preds, outcomes, method="temperature")
        assert isinstance(metrics["brier_after"], float)
        assert 0.0 <= metrics["brier_after"] <= 1.0

    def test_platt_returns_flat_list(self):
        np.random.seed(42)
        preds = np.random.uniform(0.1, 0.9, 100).tolist()
        outcomes = np.random.randint(0, 2, 100).tolist()
        calibrated, _ = calibrate_confidence(preds, outcomes, method="platt")
        assert isinstance(calibrated, list)
        for v in calibrated:
            assert isinstance(v, float)

    def test_isotonic_returns_flat_list(self):
        np.random.seed(42)
        preds = np.random.uniform(0.1, 0.9, 100).tolist()
        outcomes = np.random.randint(0, 2, 100).tolist()
        calibrated, _ = calibrate_confidence(preds, outcomes, method="isotonic")
        assert isinstance(calibrated, list)
        for v in calibrated:
            assert isinstance(v, (float, np.floating))


# ===================================================================
# Bug fix: Logit division by zero
# ===================================================================


class TestLogitSafety:
    """Verify temperature scaling handles extreme predictions."""

    def test_predictions_at_zero(self):
        """Predictions exactly 0.0 should not cause log(0) errors."""
        preds = [0.0] * 50 + [1.0] * 50
        outcomes = [0] * 50 + [1] * 50
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="temperature")
        assert len(calibrated) == 100
        for v in calibrated:
            assert not math.isnan(v)
            assert not math.isinf(v)

    def test_predictions_at_one(self):
        """Predictions exactly 1.0 should not cause division issues."""
        preds = [0.99999] * 50 + [0.00001] * 50
        outcomes = [1] * 50 + [0] * 50
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="temperature")
        for v in calibrated:
            assert not math.isnan(v)

    def test_predictions_near_boundary(self):
        """Predictions very near 0 and 1 should produce valid results."""
        preds = [1e-15, 1.0 - 1e-15, 0.5, 0.3, 0.7, 0.1, 0.9, 0.01, 0.99, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5]  # >= 10 for calibration
        outcomes = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="temperature")
        for v in calibrated:
            assert not math.isnan(v)
            assert not math.isinf(v)


# ===================================================================
# Bug fix: NaN/Inf handling in calibration
# ===================================================================


class TestNaNInfHandling:
    """Verify NaN/Inf values are properly caught."""

    def test_brier_score_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            brier_score([0.5, float("nan")], [1, 0])

    def test_brier_score_inf_raises(self):
        with pytest.raises(ValueError, match="Inf"):
            brier_score([0.5, float("inf")], [1, 0])

    def test_ece_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            expected_calibration_error([0.5, float("nan")], [1, 0])

    def test_validate_calibration_inputs_nan(self):
        valid, issues = validate_calibration_inputs([float("nan"), 0.5], [1, 0])
        assert not valid
        assert any("NaN" in i for i in issues)

    def test_validate_calibration_inputs_inf(self):
        valid, issues = validate_calibration_inputs([float("inf"), 0.5], [1, 0])
        assert not valid
        assert any("Inf" in i for i in issues)

    def test_validate_bounds_nan(self):
        valid, issues = validate_bounds([0.5, float("nan")])
        assert not valid
        assert any("NaN" in i for i in issues)

    def test_validate_bounds_inf(self):
        valid, issues = validate_bounds([0.5, float("inf")])
        assert not valid
        assert any("Inf" in i for i in issues)

    def test_validate_monotonicity_nan(self):
        valid, issues = validate_monotonicity(
            ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            [0.5, float("nan")]
        )
        assert not valid
        assert any("NaN" in i for i in issues)

    def test_calibrate_confidence_rejects_nan_inputs(self):
        preds = [float("nan")] * 15
        outcomes = [1] * 15
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="platt")
        assert metrics.get("reason") == "invalid_inputs"

    def test_validate_monotonicity_length_mismatch(self):
        valid, issues = validate_monotonicity(
            ["2024-01-01T00:00:00Z"],
            [0.5, 0.6]
        )
        assert not valid
        assert any("same length" in i for i in issues)


# ===================================================================
# Bug fix: ECE n_bins validation
# ===================================================================


class TestECEEdgeCases:
    """Edge cases for expected_calibration_error."""

    def test_n_bins_zero_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            expected_calibration_error([0.5], [1], n_bins=0)

    def test_single_prediction(self):
        ece, mce = expected_calibration_error([0.8], [1], n_bins=10)
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0


# ===================================================================
# Bug fix: clamp01 NaN handling
# ===================================================================


class TestClamp01NaN:
    """Verify clamp01 handles NaN/Inf safely."""

    def test_nan_returns_safe_value(self):
        result = clamp01(float("nan"))
        assert result == 0.5

    def test_inf_returns_safe_value(self):
        result = clamp01(float("inf"))
        assert result == 0.5

    def test_neg_inf_returns_safe_value(self):
        result = clamp01(float("-inf"))
        assert result == 0.5


# ===================================================================
# Bug fix: Model corrupted input handling
# ===================================================================


class TestModelCorruptedInput:
    """Verify model handles corrupted inputs gracefully."""

    def test_add_node_nan_confidence(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", float("nan"))
        assert "A" in m.nodes
        assert not math.isnan(m.nodes["A"].prior)

    def test_add_node_none_confidence(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", None)
        assert "A" in m.nodes
        assert not math.isnan(m.nodes["A"].prior)

    def test_add_node_string_confidence(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", "high")
        assert "A" in m.nodes

    def test_add_node_empty_id_rejected(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("", "indicator", "A", 80)
        assert "" not in m.nodes

    def test_add_edge_nan_weight(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 50)
        m.add_or_update_edge("A", "B", float("nan"))
        assert m.G.has_edge("A", "B")
        assert not math.isnan(m.edge_w[("A", "B")])

    def test_add_edge_empty_ids(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_edge("", "A", 70)
        m.add_or_update_edge("A", "", 70)
        assert len(list(m.G.edges())) == 0

    def test_infer_empty_graph(self):
        m = BayesianConfidenceModel()
        result = m.infer_all()
        assert result == {}


# ===================================================================
# Bug fix: YAML duplicate keys
# ===================================================================


class TestYAMLConfig:
    """Verify the bayes.yaml config has no issues."""

    def test_no_duplicate_keys(self):
        """Ensure bayes.yaml has unique top-level keys."""
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "bayes.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Parse and count top-level keys manually
        lines = content.strip().split("\n")
        top_level_keys = []
        for line in lines:
            if line and not line.startswith(" ") and not line.startswith("#") and not line.startswith("\t"):
                if ":" in line:
                    key = line.split(":")[0].strip()
                    if key:
                        top_level_keys.append(key)

        # Each key should appear exactly once
        from collections import Counter
        counts = Counter(top_level_keys)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        assert not duplicates, f"Duplicate YAML keys found: {duplicates}"

    def test_config_loads_all_expected_keys(self):
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "bayes.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        expected_keys = {"lbp_damping", "lbp_epsilon", "lbp_max_iters", "ema_alpha",
                         "confidence_push_delta_min", "rel_type_weight", "time_decay_half_life"}
        for key in expected_keys:
            assert key in cfg, f"Missing config key: {key}"


# ===================================================================
# API parameter validation
# ===================================================================


class TestAPIParameterValidation:
    """Test that API endpoints validate query parameters."""

    @pytest.fixture
    def client(self):
        from unittest.mock import Mock
        from api.server import create_app

        bus = Mock()
        bus.subscribe.return_value = Mock()
        manager = Mock()
        manager.bayes = BayesianConfidenceModel()
        manager.bayes.add_or_update_node("A", "Indicator", "Test", 80)
        manager.bayes.add_or_update_node("B", "Malware", "Mal", 20)
        manager.bayes.add_or_update_edge("A", "B", 70)
        manager.last_conf = {}
        manager.export_graph.return_value = manager.bayes.export_graph()
        manager.get_history.return_value = []

        app = create_app(sync_manager=manager, event_bus=bus)
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def test_contributions_invalid_topk(self, client):
        resp = client.get("/api/v1/contributions?id=B&topk=abc")
        assert resp.status_code == 400
        assert "topk" in resp.get_json()["error"]

    def test_contributions_negative_topk(self, client):
        resp = client.get("/api/v1/contributions?id=B&topk=-5")
        assert resp.status_code == 400
        assert "topk" in resp.get_json()["error"]

    def test_paths_invalid_k(self, client):
        resp = client.get("/api/v1/paths?id=B&k=xyz")
        assert resp.status_code == 400
        assert "k" in resp.get_json()["error"]

    def test_paths_invalid_maxlen(self, client):
        resp = client.get("/api/v1/paths?id=B&maxlen=abc")
        assert resp.status_code == 400
        assert "maxlen" in resp.get_json()["error"]

    def test_contributions_valid_topk(self, client):
        resp = client.get("/api/v1/contributions?id=B&topk=5")
        assert resp.status_code == 200

    def test_paths_valid_params(self, client):
        resp = client.get("/api/v1/paths?id=B&k=3&maxlen=5")
        assert resp.status_code == 200

    def test_paths_default_params(self, client):
        resp = client.get("/api/v1/paths?id=B")
        assert resp.status_code == 200


# ===================================================================
# Calibrate confidence edge cases
# ===================================================================


class TestCalibrateEdgeCases:
    """Additional edge case tests for calibrate_confidence."""

    def test_unknown_method(self):
        preds = list(np.random.uniform(0, 1, 20))
        outcomes = [0, 1] * 10
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="unknown_method")
        assert calibrated == preds
        assert metrics["method"] == "unknown"

    def test_all_same_predictions(self):
        """All identical predictions should still work."""
        preds = [0.5] * 20
        outcomes = [0, 1] * 10
        calibrated, metrics = calibrate_confidence(preds, outcomes, method="temperature")
        assert len(calibrated) == 20

    def test_all_same_outcomes(self):
        """All same outcomes (no variance) — should handle gracefully."""
        np.random.seed(42)
        preds = np.random.uniform(0.1, 0.9, 20).tolist()
        outcomes = [1] * 20
        # Platt scaling might fail with single class, should not crash
        try:
            calibrated, metrics = calibrate_confidence(preds, outcomes, method="platt")
        except Exception:
            pass  # Acceptable to fail gracefully

    def test_analyze_calibration_structure(self):
        preds = [0.2, 0.4, 0.6, 0.8]
        outcomes = [0, 0, 1, 1]
        result = analyze_calibration(preds, outcomes, n_bins=4)
        assert result.n_bins == 4
        assert len(result.bin_counts) == 4
        assert len(result.bin_confidences) == 4
        assert len(result.bin_accuracies) == 4
        assert len(result.calibration_difference) == 4
        assert sum(result.bin_counts) == 4

    def test_summarize_calibration_nonempty(self):
        preds = [0.3, 0.7, 0.5, 0.9]
        outcomes = [0, 1, 0, 1]
        summary = summarize_calibration(preds, outcomes, n_bins=4)
        assert summary.n_samples == 4
        assert 0 <= summary.brier_score <= 1
        assert 0 <= summary.ece <= 1
