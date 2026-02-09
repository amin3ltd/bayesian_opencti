"""
Comprehensive tests for the data validation module.

Tests STIX schema validation, referential integrity, graph invariant
checks, config validation, model output validation, and SyncManager
state validation.
"""

import json
import math
from pathlib import Path

import pytest
import yaml

from bayes.model import BayesianConfidenceModel, clamp01
from service.eventbus import EventBus
from service.sync_manager import SyncManager
from validation.data_validators import (
    ValidationReport,
    validate_stix_objects,
    validate_referential_integrity,
    validate_graph_invariants,
    validate_config,
    validate_inference_results,
    validate_sync_manager_state,
    run_full_validation,
)

pytestmark = pytest.mark.unit


# ===================================================================
# Helper factories
# ===================================================================


def _stix_obj(oid, otype, name="test", confidence=50, **extra):
    obj = {
        "id": oid,
        "type": otype,
        "name": name,
        "spec_version": "2.1",
        "created": "2024-01-01T00:00:00Z",
        "modified": "2024-01-01T00:00:00Z",
        "confidence": confidence,
    }
    obj.update(extra)
    return obj


def _stix_rel(rid, src, dst, rel_type="uses", confidence=70):
    return {
        "id": rid,
        "type": "relationship",
        "spec_version": "2.1",
        "created": "2024-01-01T00:00:00Z",
        "modified": "2024-01-01T00:00:00Z",
        "relationship_type": rel_type,
        "source_ref": src,
        "target_ref": dst,
        "confidence": confidence,
    }


def _make_manager(objects=None, rels=None, cfg=None, max_parents=5):
    bus = EventBus()
    cfg = cfg or {"ema_alpha": 0.0, "confidence_push_delta_min": 0}
    m = SyncManager(max_parents=max_parents, bus=bus, cfg=cfg)
    if objects:
        m.build_from_opencti(objects, rels or [])
    return m


# ===================================================================
# STIX Schema Validation
# ===================================================================


class TestSTIXValidation:
    """Tests for STIX object schema validation."""

    def test_valid_objects(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator"),
            _stix_obj("malware--bbbbbbbb-1111-2222-3333-444444444444", "malware"),
        ]
        report = validate_stix_objects(objects)
        assert report.is_valid

    def test_missing_id(self):
        objects = [{"type": "indicator", "name": "no-id"}]
        report = validate_stix_objects(objects)
        assert not report.is_valid
        assert any("Missing required field 'id'" in str(i) for i in report.errors)

    def test_missing_type(self):
        objects = [{"id": "indicator--aaaaaaaa-1111-2222-3333-444444444444", "name": "no-type"}]
        report = validate_stix_objects(objects)
        assert not report.is_valid
        assert any("Missing required field 'type'" in str(i) for i in report.errors)

    def test_duplicate_ids(self):
        oid = "indicator--aaaaaaaa-1111-2222-3333-444444444444"
        objects = [
            _stix_obj(oid, "indicator", name="first"),
            _stix_obj(oid, "indicator", name="second"),
        ]
        report = validate_stix_objects(objects)
        assert report.warning_count >= 1
        assert any("Duplicate" in str(w) for w in report.warnings)

    def test_id_prefix_mismatch(self):
        objects = [_stix_obj("malware--aaaaaaaa-1111-2222-3333-444444444444", "indicator")]
        report = validate_stix_objects(objects)
        assert any("prefix" in str(w) for w in report.warnings)

    def test_unknown_type(self):
        objects = [_stix_obj("foo--aaaaaaaa-1111-2222-3333-444444444444", "foo")]
        report = validate_stix_objects(objects)
        assert any("Unknown STIX type" in str(w) for w in report.warnings)

    def test_confidence_out_of_range(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator", confidence=150)
        ]
        report = validate_stix_objects(objects)
        assert not report.is_valid
        assert any("out of range" in str(e) for e in report.errors)

    def test_confidence_negative(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator", confidence=-10)
        ]
        report = validate_stix_objects(objects)
        assert not report.is_valid

    def test_confidence_nan(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator", confidence=float("nan"))
        ]
        report = validate_stix_objects(objects)
        assert not report.is_valid
        assert any("NaN" in str(e) for e in report.errors)

    def test_confidence_non_numeric(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator", confidence="high")
        ]
        report = validate_stix_objects(objects)
        assert not report.is_valid
        assert any("numeric" in str(e) for e in report.errors)

    def test_confidence_none_ok(self):
        """None confidence should NOT produce an error (it means not provided)."""
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator", confidence=None)
        ]
        report = validate_stix_objects(objects)
        assert report.is_valid

    def test_timestamp_created_after_modified(self):
        objects = [{
            "id": "indicator--aaaaaaaa-1111-2222-3333-444444444444",
            "type": "indicator",
            "created": "2025-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
        }]
        report = validate_stix_objects(objects)
        assert any("created" in str(w) and "modified" in str(w) for w in report.warnings)

    def test_unparseable_timestamp(self):
        objects = [{
            "id": "indicator--aaaaaaaa-1111-2222-3333-444444444444",
            "type": "indicator",
            "created": "not-a-date",
        }]
        report = validate_stix_objects(objects)
        assert any("Unparseable" in str(e) for e in report.errors)

    def test_relationship_missing_source_ref(self):
        objects = [{
            "id": "relationship--aaaaaaaa-1111-2222-3333-444444444444",
            "type": "relationship",
            "target_ref": "malware--bbbbbbbb-1111-2222-3333-444444444444",
            "relationship_type": "uses",
        }]
        report = validate_stix_objects(objects)
        assert not report.is_valid
        assert any("source_ref" in str(e) for e in report.errors)

    def test_relationship_missing_target_ref(self):
        objects = [{
            "id": "relationship--aaaaaaaa-1111-2222-3333-444444444444",
            "type": "relationship",
            "source_ref": "indicator--bbbbbbbb-1111-2222-3333-444444444444",
            "relationship_type": "uses",
        }]
        report = validate_stix_objects(objects)
        assert not report.is_valid
        assert any("target_ref" in str(e) for e in report.errors)

    def test_self_referencing_relationship(self):
        oid = "malware--aaaaaaaa-1111-2222-3333-444444444444"
        objects = [{
            "id": "relationship--bbbbbbbb-1111-2222-3333-444444444444",
            "type": "relationship",
            "source_ref": oid,
            "target_ref": oid,
            "relationship_type": "uses",
        }]
        report = validate_stix_objects(objects)
        assert any("Self-referencing" in str(w) for w in report.warnings)

    def test_indicator_missing_pattern(self):
        objects = [{
            "id": "indicator--aaaaaaaa-1111-2222-3333-444444444444",
            "type": "indicator",
        }]
        report = validate_stix_objects(objects)
        assert any("pattern" in str(w) for w in report.warnings)


# ===================================================================
# Referential Integrity
# ===================================================================


class TestReferentialIntegrity:
    """Tests for referential integrity validation."""

    def test_valid_references(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator"),
            _stix_obj("malware--bbbbbbbb-1111-2222-3333-444444444444", "malware"),
            _stix_rel("relationship--cccccccc-1111-2222-3333-444444444444",
                       "indicator--aaaaaaaa-1111-2222-3333-444444444444",
                       "malware--bbbbbbbb-1111-2222-3333-444444444444"),
        ]
        report = validate_referential_integrity(objects)
        assert report.is_valid

    def test_missing_source_ref(self):
        objects = [
            _stix_obj("malware--bbbbbbbb-1111-2222-3333-444444444444", "malware"),
            _stix_rel("relationship--cccccccc-1111-2222-3333-444444444444",
                       "indicator--missing-1111-2222-3333-444444444444",
                       "malware--bbbbbbbb-1111-2222-3333-444444444444"),
        ]
        report = validate_referential_integrity(objects)
        assert report.warning_count >= 1
        assert any("source_ref" in str(w) and "not found" in str(w) for w in report.warnings)

    def test_missing_target_ref(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator"),
            _stix_rel("relationship--cccccccc-1111-2222-3333-444444444444",
                       "indicator--aaaaaaaa-1111-2222-3333-444444444444",
                       "malware--missing-1111-2222-3333-444444444444"),
        ]
        report = validate_referential_integrity(objects)
        assert report.warning_count >= 1

    def test_report_object_refs_missing(self):
        objects = [{
            "id": "report--aaaaaaaa-1111-2222-3333-444444444444",
            "type": "report",
            "object_refs": ["malware--missing-1111-2222-3333-444444444444"],
        }]
        report = validate_referential_integrity(objects)
        assert report.warning_count >= 1
        assert any("object_ref" in str(w) for w in report.warnings)

    def test_sighting_refs_missing(self):
        objects = [{
            "id": "sighting--aaaaaaaa-1111-2222-3333-444444444444",
            "type": "sighting",
            "sighting_of_ref": "indicator--missing-1111-2222-3333-444444444444",
        }]
        report = validate_referential_integrity(objects)
        assert report.warning_count >= 1


# ===================================================================
# Graph Invariant Validation
# ===================================================================


class TestGraphInvariants:
    """Tests for graph invariant validation."""

    def test_valid_graph(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 50)
        m.add_or_update_edge("A", "B", 70)
        m.infer_all()
        report = validate_graph_invariants(m)
        assert report.is_valid

    def test_empty_graph_warning(self):
        m = BayesianConfidenceModel()
        report = validate_graph_invariants(m)
        assert report.warning_count >= 1
        assert any("no nodes" in str(w) for w in report.warnings)

    def test_edge_weight_out_of_range_detected(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 50)
        m.add_or_update_edge("A", "B", 70)
        # Manually corrupt an edge weight
        m.edge_w[("A", "B")] = 1.5
        report = validate_graph_invariants(m)
        assert not report.is_valid
        assert any("out of [0,1]" in str(e) for e in report.errors)

    def test_nan_edge_weight_detected(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 50)
        m.add_or_update_edge("A", "B", 70)
        m.edge_w[("A", "B")] = float("nan")
        report = validate_graph_invariants(m)
        assert not report.is_valid
        assert any("NaN" in str(e) for e in report.errors)

    def test_nan_belief_detected(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.infer_all()
        m.nodes["A"].belief = float("nan")
        report = validate_graph_invariants(m)
        assert not report.is_valid

    def test_belief_out_of_range_detected(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.infer_all()
        m.nodes["A"].belief = 1.5
        report = validate_graph_invariants(m)
        assert not report.is_valid

    def test_parent_cap_violation_detected(self):
        m = BayesianConfidenceModel(max_parents=2)
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "indicator", "B", 70)
        m.add_or_update_node("C", "indicator", "C", 60)
        m.add_or_update_node("D", "malware", "D", 50)
        m.add_or_update_edge("A", "D", 80)
        m.add_or_update_edge("B", "D", 70)
        # Manually add a third edge bypassing the cap
        m.G.add_edge("C", "D")
        m.edge_w[("C", "D")] = 0.5
        report = validate_graph_invariants(m)
        assert report.warning_count >= 1
        assert any("parents" in str(w) and "exceeding" in str(w) for w in report.warnings)


# ===================================================================
# Config Validation
# ===================================================================


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config(self):
        cfg = {
            "lbp_damping": 0.55,
            "lbp_epsilon": 1e-4,
            "lbp_max_iters": 100,
            "ema_alpha": 0.35,
            "confidence_push_delta_min": 2,
            "default_rel_weight": 0.5,
            "rel_type_weight": {"indicates": 0.85, "uses": 0.35},
            "time_decay_half_life": {"Indicator": 60},
        }
        report = validate_config(cfg)
        assert report.is_valid

    def test_damping_out_of_range(self):
        cfg = {"lbp_damping": 1.5}
        report = validate_config(cfg)
        assert report.warning_count >= 1

    def test_damping_zero(self):
        cfg = {"lbp_damping": 0.0}
        report = validate_config(cfg)
        assert report.warning_count >= 1

    def test_ema_alpha_out_of_range(self):
        cfg = {"ema_alpha": 2.0}
        report = validate_config(cfg)
        assert report.warning_count >= 1

    def test_negative_epsilon(self):
        cfg = {"lbp_epsilon": -0.001}
        report = validate_config(cfg)
        assert report.warning_count >= 1

    def test_nan_in_config(self):
        cfg = {"lbp_damping": float("nan")}
        report = validate_config(cfg)
        assert not report.is_valid

    def test_non_numeric_config(self):
        cfg = {"lbp_damping": "fast"}
        report = validate_config(cfg)
        assert not report.is_valid

    def test_rel_type_weight_out_of_range(self):
        cfg = {"rel_type_weight": {"uses": 1.5}}
        report = validate_config(cfg)
        assert report.warning_count >= 1

    def test_time_decay_negative(self):
        cfg = {"time_decay_half_life": {"Indicator": -10}}
        report = validate_config(cfg)
        assert report.warning_count >= 1

    def test_empty_config_valid(self):
        report = validate_config({})
        assert report.is_valid  # All optional, no errors

    def test_actual_config_file(self):
        """Validate the actual bayes.yaml config."""
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "bayes.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        report = validate_config(cfg)
        assert report.is_valid, f"Config validation failed: {report}"


# ===================================================================
# Model Output Validation
# ===================================================================


class TestModelOutputValidation:
    """Tests for inference result validation."""

    def test_valid_inference(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 50)
        m.add_or_update_edge("A", "B", 70)
        beliefs = m.infer_all()
        report = validate_inference_results(m, beliefs)
        assert report.is_valid

    def test_missing_belief(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 50)
        m.infer_all()
        report = validate_inference_results(m, {"A": 0.8})  # B missing
        assert not report.is_valid
        assert any("Missing belief" in str(e) for e in report.errors)

    def test_nan_belief_detected(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.infer_all()
        report = validate_inference_results(m, {"A": float("nan")})
        assert not report.is_valid

    def test_out_of_range_belief(self):
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.infer_all()
        report = validate_inference_results(m, {"A": 1.5})
        assert not report.is_valid

    def test_noisy_or_monotonicity_holds(self):
        """Verify posterior >= prior for acyclic node with positive parents."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 20)
        m.add_or_update_edge("A", "B", 70)
        beliefs = m.infer_all()
        report = validate_inference_results(m, beliefs)
        assert report.is_valid
        assert beliefs["B"] >= m.nodes["B"].prior - 1e-6

    def test_convergence_info_populated(self):
        """Cyclic network should produce convergence info."""
        m = BayesianConfidenceModel(damping=0.55, max_iters=1000)
        m.add_or_update_node("A", "indicator", "A", 50)
        m.add_or_update_node("B", "malware", "B", 50)
        m.add_or_update_edge("A", "B", 60)
        m.add_or_update_edge("B", "A", 60)
        m.infer_all()
        info = m.get_convergence_info()
        assert len(info) >= 1
        assert info[0]["converged"] is True
        assert info[0]["scc_size"] == 2

    def test_convergence_info_empty_for_dag(self):
        """DAG should have no cyclic convergence info."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "indicator", "A", 80)
        m.add_or_update_node("B", "malware", "B", 50)
        m.add_or_update_edge("A", "B", 70)
        m.infer_all()
        info = m.get_convergence_info()
        assert len(info) == 0


# ===================================================================
# SyncManager State Validation
# ===================================================================


class TestSyncManagerValidation:
    """Tests for SyncManager state validation."""

    def test_valid_state(self):
        objects = [
            {"id": "A", "type": "indicator", "name": "A", "confidence": 80},
            {"id": "B", "type": "malware", "name": "B", "confidence": 50},
        ]
        rels = [{"source_ref": "A", "target_ref": "B", "type": "uses", "confidence": 70}]
        m = _make_manager(objects, rels)
        m.run_inference_and_diff()
        report = validate_sync_manager_state(m)
        assert report.is_valid

    def test_last_conf_out_of_range(self):
        m = _make_manager([{"id": "A", "type": "indicator", "name": "A", "confidence": 80}])
        m.run_inference_and_diff()
        m.last_conf["A"] = 150  # Corrupt
        report = validate_sync_manager_state(m)
        assert not report.is_valid

    def test_no_belief_warning(self):
        m = _make_manager([{"id": "A", "type": "indicator", "name": "A", "confidence": 80}])
        # Don't run inference — belief stays None
        report = validate_sync_manager_state(m)
        assert report.warning_count >= 1


# ===================================================================
# Full Pipeline Validation
# ===================================================================


class TestFullValidationPipeline:
    """Tests for the full validation pipeline."""

    def test_full_pipeline_valid(self):
        objects = [
            _stix_obj("indicator--aaaaaaaa-1111-2222-3333-444444444444", "indicator"),
            _stix_obj("malware--bbbbbbbb-1111-2222-3333-444444444444", "malware"),
            _stix_rel("relationship--cccccccc-1111-2222-3333-444444444444",
                       "indicator--aaaaaaaa-1111-2222-3333-444444444444",
                       "malware--bbbbbbbb-1111-2222-3333-444444444444"),
        ]
        cfg = {"lbp_damping": 0.55, "ema_alpha": 0.0, "confidence_push_delta_min": 0}
        m = _make_manager(
            [o for o in objects if o["type"] != "relationship"],
            [o for o in objects if o["type"] == "relationship"],
            cfg=cfg,
        )
        beliefs = m.bayes.infer_all()
        m.run_inference_and_diff()
        report = run_full_validation(objects, cfg, sync_manager=m, beliefs=beliefs)
        # Should have no errors (warnings are acceptable)
        assert report.error_count == 0, f"Full validation errors: {report}"

    def test_full_pipeline_with_bad_data(self):
        objects = [
            {"type": "indicator"},  # Missing ID
            _stix_obj("malware--bbbbbbbb-1111-2222-3333-444444444444", "malware", confidence=200),
        ]
        cfg = {"lbp_damping": 5.0}  # Out of range
        report = run_full_validation(objects, cfg)
        assert report.error_count >= 1  # Multiple errors expected

    def test_full_pipeline_with_sample_data(self):
        """Run full validation on the actual sample_data.json."""
        root = Path(__file__).resolve().parents[1]
        data_path = root / "sample_data.json"
        cfg_path = root / "config" / "bayes.yaml"

        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        objects = data.get("objects", [])

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        non_rels = [o for o in objects if o.get("type") not in ("relationship", "sighting")]
        rels = [o for o in objects if o.get("type") == "relationship"]

        m = _make_manager(non_rels, rels, cfg=cfg)
        beliefs = m.bayes.infer_all()
        m.run_inference_and_diff()

        report = run_full_validation(objects, cfg, sync_manager=m, beliefs=beliefs)
        # Real sample data should have zero errors
        assert report.error_count == 0, f"Sample data validation errors:\n{report}"


# ===================================================================
# ValidationReport API
# ===================================================================


class TestValidationReport:
    """Tests for the ValidationReport data structure."""

    def test_empty_report(self):
        r = ValidationReport()
        assert r.is_valid
        assert r.error_count == 0
        assert r.warning_count == 0

    def test_add_error(self):
        r = ValidationReport()
        r.add("error", "test", "Something bad")
        assert not r.is_valid
        assert r.error_count == 1

    def test_add_warning(self):
        r = ValidationReport()
        r.add("warning", "test", "Something concerning")
        assert r.is_valid  # Warnings don't make report invalid
        assert r.warning_count == 1

    def test_summary(self):
        r = ValidationReport()
        r.add("error", "stix", "err1")
        r.add("warning", "graph", "warn1")
        r.add("error", "stix", "err2")
        s = r.summary()
        assert s["total_issues"] == 3
        assert s["errors"] == 2
        assert s["warnings"] == 1
        assert s["is_valid"] is False
        assert s["by_category"]["stix"] == 2
        assert s["by_category"]["graph"] == 1

    def test_str_representation(self):
        r = ValidationReport()
        r.add("error", "test", "Something bad", object_id="X", field_name="f")
        s = str(r)
        assert "error" in s.lower()
        assert "X" in s
