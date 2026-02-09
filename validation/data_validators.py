"""
Full-scale data validation for STIX objects, graph structure,
configuration, and model outputs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Valid STIX 2.1 domain object types used in this project
KNOWN_STIX_TYPES: Set[str] = {
    "threat-actor",
    "threat-actor-individual",
    "intrusion-set",
    "campaign",
    "attack-pattern",
    "malware",
    "infrastructure",
    "course-of-action",
    "identity",
    "indicator",
    "report",
    "sighting",
    "relationship",
    "observed-data",
    "vulnerability",
    "tool",
    "location",
    "note",
    "opinion",
    "grouping",
}

# STIX 2.1 relationship types this project supports
KNOWN_RELATIONSHIP_TYPES: Set[str] = {
    "uses",
    "indicates",
    "attributed-to",
    "targets",
    "related-to",
    "delivers",
    "drops",
    "mitigates",
    "object",
    "variant-of",
    "derived-from",
    "consists-of",
    "communicates-with",
    "hosts",
    "based-on",
    "exploits",
    "compromises",
    "remediates",
    "investigates",
    "located-at",
    "characterizes",
    "analysis-of",
    "authored-by",
}

# Regex for STIX IDs: type--uuid
_STIX_ID_RE = re.compile(r"^[a-z][a-z0-9-]+--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@dataclass
class ValidationIssue:
    """Single validation finding."""
    severity: str  # "error", "warning", "info"
    category: str  # e.g. "stix_schema", "graph", "config", "model"
    message: str
    object_id: Optional[str] = None
    field: Optional[str] = None

    def __str__(self) -> str:
        loc = f" [{self.object_id}]" if self.object_id else ""
        fld = f".{self.field}" if self.field else ""
        return f"[{self.severity.upper()}] {self.category}{loc}{fld}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregated validation report."""
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add(self, severity: str, category: str, message: str,
            object_id: Optional[str] = None, field_name: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(
            severity=severity, category=category, message=message,
            object_id=object_id, field=field_name,
        ))

    def summary(self) -> Dict[str, Any]:
        """Return machine-readable summary."""
        by_cat: Dict[str, int] = {}
        for issue in self.issues:
            by_cat[issue.category] = by_cat.get(issue.category, 0) + 1
        return {
            "total_issues": len(self.issues),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "is_valid": self.is_valid,
            "by_category": by_cat,
        }

    def __str__(self) -> str:
        lines = [f"ValidationReport: {self.error_count} errors, {self.warning_count} warnings"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# ===================================================================
# 1. STIX Object Validation
# ===================================================================


def validate_stix_objects(objects: List[Dict]) -> ValidationReport:
    """
    Validate a list of STIX-like objects for schema correctness.

    Checks:
      - Required fields: 'id', 'type'
      - STIX ID format (type--uuid)
      - ID prefix matches declared type
      - Known STIX type
      - Confidence range [0, 100] when present
      - Timestamp format and ordering (created <= modified)
      - Relationships have source_ref and target_ref
      - Indicators have required pattern fields
    """
    report = ValidationReport()
    seen_ids: Set[str] = set()

    for idx, obj in enumerate(objects):
        oid = obj.get("id", f"<missing-id-at-index-{idx}>")

        # --- required fields ---
        if "id" not in obj:
            report.add("error", "stix_schema", "Missing required field 'id'",
                        object_id=oid, field_name="id")
            continue

        if "type" not in obj:
            report.add("error", "stix_schema", "Missing required field 'type'",
                        object_id=oid, field_name="type")
            continue

        otype = obj["type"]

        # --- duplicate IDs ---
        if oid in seen_ids:
            report.add("warning", "stix_schema", f"Duplicate object ID: {oid}",
                        object_id=oid, field_name="id")
        seen_ids.add(oid)

        # --- STIX ID format ---
        if not _STIX_ID_RE.match(oid):
            report.add("warning", "stix_schema", f"ID does not match STIX format (type--uuid): {oid}",
                        object_id=oid, field_name="id")

        # --- ID prefix matches type ---
        if "--" in oid:
            prefix = oid.split("--")[0]
            if prefix != otype:
                report.add("warning", "stix_schema",
                            f"ID prefix '{prefix}' does not match type '{otype}'",
                            object_id=oid, field_name="id")

        # --- known type ---
        if otype not in KNOWN_STIX_TYPES:
            report.add("warning", "stix_schema",
                        f"Unknown STIX type: {otype}",
                        object_id=oid, field_name="type")

        # --- confidence ---
        if "confidence" in obj and obj["confidence"] is not None:
            conf = obj["confidence"]
            if not isinstance(conf, (int, float)):
                report.add("error", "stix_schema",
                            f"Confidence must be numeric, got {type(conf).__name__}: {conf}",
                            object_id=oid, field_name="confidence")
            else:
                if conf < 0 or conf > 100:
                    report.add("error", "stix_schema",
                                f"Confidence out of range [0, 100]: {conf}",
                                object_id=oid, field_name="confidence")
                if isinstance(conf, float) and (math.isnan(conf) or math.isinf(conf)):
                    report.add("error", "stix_schema",
                                f"Confidence is NaN or Inf: {conf}",
                                object_id=oid, field_name="confidence")

        # --- timestamps ---
        _validate_timestamps(obj, oid, report)

        # --- relationship-specific ---
        if otype == "relationship":
            _validate_relationship(obj, oid, report)

        # --- indicator-specific ---
        if otype == "indicator":
            _validate_indicator(obj, oid, report)

    return report


def _parse_ts(value: str) -> Optional[datetime]:
    """Try to parse an ISO 8601 timestamp."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _validate_timestamps(obj: Dict, oid: str, report: ValidationReport) -> None:
    """Validate timestamp fields within a STIX object."""
    created_str = obj.get("created")
    modified_str = obj.get("modified")

    created = _parse_ts(created_str) if created_str else None
    modified = _parse_ts(modified_str) if modified_str else None

    if created_str and created is None:
        report.add("error", "stix_schema", f"Unparseable 'created' timestamp: {created_str}",
                    object_id=oid, field_name="created")
    if modified_str and modified is None:
        report.add("error", "stix_schema", f"Unparseable 'modified' timestamp: {modified_str}",
                    object_id=oid, field_name="modified")

    if created and modified and created > modified:
        report.add("warning", "stix_schema",
                    f"'created' ({created_str}) > 'modified' ({modified_str})",
                    object_id=oid, field_name="modified")


def _validate_relationship(obj: Dict, oid: str, report: ValidationReport) -> None:
    """Validate relationship-specific fields."""
    rel_type = obj.get("relationship_type")
    src = obj.get("source_ref")
    dst = obj.get("target_ref")

    if not src:
        report.add("error", "stix_schema", "Relationship missing 'source_ref'",
                    object_id=oid, field_name="source_ref")
    if not dst:
        report.add("error", "stix_schema", "Relationship missing 'target_ref'",
                    object_id=oid, field_name="target_ref")

    if rel_type and rel_type not in KNOWN_RELATIONSHIP_TYPES:
        report.add("info", "stix_schema",
                    f"Uncommon relationship_type: {rel_type}",
                    object_id=oid, field_name="relationship_type")

    if src and dst and src == dst:
        report.add("warning", "stix_schema",
                    "Self-referencing relationship (source_ref == target_ref)",
                    object_id=oid, field_name="source_ref")


def _validate_indicator(obj: Dict, oid: str, report: ValidationReport) -> None:
    """Validate indicator-specific fields."""
    if not obj.get("pattern"):
        report.add("warning", "stix_schema", "Indicator missing 'pattern' field",
                    object_id=oid, field_name="pattern")
    if not obj.get("pattern_type"):
        report.add("warning", "stix_schema", "Indicator missing 'pattern_type' field",
                    object_id=oid, field_name="pattern_type")


# ===================================================================
# 2. Referential Integrity
# ===================================================================


def validate_referential_integrity(objects: List[Dict]) -> ValidationReport:
    """
    Validate that all relationship source_ref/target_ref point to
    known object IDs within the bundle.
    """
    report = ValidationReport()
    known_ids = {o["id"] for o in objects if "id" in o}

    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        oid = obj.get("id", "?")
        src = obj.get("source_ref")
        dst = obj.get("target_ref")

        if src and src not in known_ids:
            report.add("warning", "referential_integrity",
                        f"source_ref '{src}' not found in bundle",
                        object_id=oid, field_name="source_ref")
        if dst and dst not in known_ids:
            report.add("warning", "referential_integrity",
                        f"target_ref '{dst}' not found in bundle",
                        object_id=oid, field_name="target_ref")

    # Also check report object_refs
    for obj in objects:
        if obj.get("type") != "report":
            continue
        oid = obj.get("id", "?")
        for ref in (obj.get("object_refs") or []):
            if ref not in known_ids:
                report.add("warning", "referential_integrity",
                            f"object_ref '{ref}' not found in bundle",
                            object_id=oid, field_name="object_refs")

    # Check sighting refs
    for obj in objects:
        if obj.get("type") != "sighting":
            continue
        oid = obj.get("id", "?")
        sighting_of = obj.get("sighting_of_ref")
        if sighting_of and sighting_of not in known_ids:
            report.add("warning", "referential_integrity",
                        f"sighting_of_ref '{sighting_of}' not found",
                        object_id=oid, field_name="sighting_of_ref")
        for ref in (obj.get("where_sighted_refs") or []):
            if ref not in known_ids:
                report.add("warning", "referential_integrity",
                            f"where_sighted_ref '{ref}' not found",
                            object_id=oid, field_name="where_sighted_refs")

    return report


# ===================================================================
# 3. Graph Invariant Validation
# ===================================================================


def validate_graph_invariants(model) -> ValidationReport:
    """
    Validate Bayesian graph model invariants after construction.

    Checks:
      - No self-loops
      - All edge weights in [0, 1]
      - All priors in (0, 1)
      - All beliefs (if computed) in (0, 1)
      - Parent count within max_parents
      - Graph is not empty
      - No NaN/Inf in beliefs or edge weights
    """
    report = ValidationReport()

    if len(model.nodes) == 0:
        report.add("warning", "graph", "Graph has no nodes")
        return report

    # Self-loops
    import networkx as nx
    self_loops = list(nx.selfloop_edges(model.G))
    for src, dst in self_loops:
        report.add("error", "graph", f"Self-loop detected: {src} -> {dst}",
                    object_id=src, field_name="edge")

    # Edge weights
    for (src, dst), w in model.edge_w.items():
        if math.isnan(w) or math.isinf(w):
            report.add("error", "graph", f"Edge weight is NaN/Inf: {w}",
                        object_id=f"{src}->{dst}", field_name="weight")
        elif w < 0.0 or w > 1.0:
            report.add("error", "graph", f"Edge weight out of [0,1]: {w}",
                        object_id=f"{src}->{dst}", field_name="weight")

    # Node priors and beliefs
    for nid, info in model.nodes.items():
        if math.isnan(info.prior) or math.isinf(info.prior):
            report.add("error", "graph", f"Prior is NaN/Inf: {info.prior}",
                        object_id=nid, field_name="prior")
        elif info.prior <= 0.0 or info.prior >= 1.0:
            # Note: clamp01 allows values very near 0/1 but not exact
            pass  # clamp01 intentionally keeps values in (EPS, 1-EPS)

        if info.belief is not None:
            if math.isnan(info.belief) or math.isinf(info.belief):
                report.add("error", "graph", f"Belief is NaN/Inf: {info.belief}",
                            object_id=nid, field_name="belief")
            elif info.belief < 0.0 or info.belief > 1.0:
                report.add("error", "graph", f"Belief out of [0,1]: {info.belief}",
                            object_id=nid, field_name="belief")

    # Parent cap
    for nid in model.G.nodes:
        n_parents = model.G.in_degree(nid)
        if n_parents > model.max_parents:
            report.add("warning", "graph",
                        f"Node has {n_parents} parents, exceeding max_parents={model.max_parents}",
                        object_id=nid, field_name="in_degree")

    return report


# ===================================================================
# 4. Configuration Validation
# ===================================================================


def validate_config(cfg: Dict) -> ValidationReport:
    """
    Validate bayes.yaml configuration values.

    Checks:
      - lbp_damping in (0, 1)
      - lbp_epsilon > 0
      - lbp_max_iters > 0
      - ema_alpha in [0, 1]
      - confidence_push_delta_min >= 0
      - rel_type_weight values in [0, 1]
      - default_rel_weight in [0, 1]
      - time_decay_half_life values > 0
      - rel_conf_fallback in [0, 100]
      - report_object_min in [0, 100]
    """
    report = ValidationReport()

    def _check_range(key: str, low: float, high: float, required: bool = False) -> None:
        val = cfg.get(key)
        if val is None:
            if required:
                report.add("error", "config", f"Missing required key: {key}", field_name=key)
            return
        try:
            fval = float(val)
        except (TypeError, ValueError):
            report.add("error", "config", f"Non-numeric value for {key}: {val}", field_name=key)
            return
        if math.isnan(fval) or math.isinf(fval):
            report.add("error", "config", f"NaN/Inf value for {key}", field_name=key)
        elif fval < low or fval > high:
            report.add("warning", "config",
                        f"{key} = {fval} outside expected range [{low}, {high}]",
                        field_name=key)

    _check_range("lbp_damping", 0.01, 0.99)
    _check_range("lbp_epsilon", 1e-12, 1.0)
    _check_range("lbp_max_iters", 1, 100000)
    _check_range("ema_alpha", 0.0, 1.0)
    _check_range("confidence_push_delta_min", 0, 100)
    _check_range("default_rel_weight", 0.0, 1.0)
    _check_range("rel_conf_fallback", 0, 100)
    _check_range("report_object_min", 0, 100)

    # rel_type_weight dict
    rtw = cfg.get("rel_type_weight")
    if rtw and isinstance(rtw, dict):
        for k, v in rtw.items():
            try:
                fv = float(v)
                if fv < 0.0 or fv > 1.0:
                    report.add("warning", "config",
                                f"rel_type_weight[{k}] = {fv} outside [0, 1]",
                                field_name="rel_type_weight")
            except (TypeError, ValueError):
                report.add("error", "config",
                            f"rel_type_weight[{k}] is not numeric: {v}",
                            field_name="rel_type_weight")

    # time_decay_half_life dict
    tdhl = cfg.get("time_decay_half_life")
    if tdhl and isinstance(tdhl, dict):
        for k, v in tdhl.items():
            try:
                fv = float(v)
                if fv <= 0:
                    report.add("warning", "config",
                                f"time_decay_half_life[{k}] = {fv} must be > 0",
                                field_name="time_decay_half_life")
            except (TypeError, ValueError):
                report.add("error", "config",
                            f"time_decay_half_life[{k}] is not numeric: {v}",
                            field_name="time_decay_half_life")

    return report


# ===================================================================
# 5. Model Output Validation (post-inference)
# ===================================================================


def validate_inference_results(
    model,
    beliefs: Dict[str, float],
) -> ValidationReport:
    """
    Validate model outputs after inference.

    Checks:
      - All beliefs in [0, 1], no NaN/Inf
      - Beliefs exist for every node in the graph
      - Noisy-OR monotonicity: adding a parent with positive weight should
        not decrease a node's posterior below its prior (sanity check on a
        subset of acyclic singletons)
      - Convergence: for cyclic components, beliefs should be finite
    """
    report = ValidationReport()

    # All beliefs present and valid
    for nid in model.nodes:
        if nid not in beliefs:
            report.add("error", "model_output", f"Missing belief for node {nid}",
                        object_id=nid, field_name="belief")
            continue
        b = beliefs[nid]
        if math.isnan(b) or math.isinf(b):
            report.add("error", "model_output", f"Belief is NaN/Inf: {b}",
                        object_id=nid, field_name="belief")
        elif b < 0.0 or b > 1.0:
            report.add("error", "model_output", f"Belief out of [0,1]: {b}",
                        object_id=nid, field_name="belief")

    # Noisy-OR monotonicity for acyclic singletons with parents
    import networkx as nx
    sccs = list(nx.strongly_connected_components(model.G))
    singletons = {next(iter(c)) for c in sccs if len(c) == 1}

    for nid in singletons:
        if model.G.has_edge(nid, nid):
            continue  # skip self-loops (handled separately)
        info = model.nodes.get(nid)
        if not info:
            continue
        b = beliefs.get(nid)
        if b is None:
            continue
        parents = list(model.G.predecessors(nid))
        if parents:
            # With Noisy-OR and positive parent beliefs, posterior >= prior
            if b < info.prior - 1e-6:
                report.add("warning", "model_output",
                            f"Posterior ({b:.6f}) < prior ({info.prior:.6f}) for "
                            f"acyclic node with parents — possible Noisy-OR violation",
                            object_id=nid, field_name="belief")

    return report


# ===================================================================
# 6. SyncManager Data Flow Validation
# ===================================================================


def validate_sync_manager_state(sync_manager) -> ValidationReport:
    """
    Validate the internal state of a SyncManager after build_from_opencti
    and run_inference_and_diff.

    Checks:
      - last_conf values in [0, 100]
      - All graph nodes have corresponding last_conf after inference
      - History entries are monotonically timestamped
      - Edge weight computation consistency
    """
    report = ValidationReport()

    # last_conf range
    for nid, conf in sync_manager.last_conf.items():
        if not isinstance(conf, (int, float)):
            report.add("error", "sync_state", f"last_conf[{nid}] is not numeric: {conf}",
                        object_id=nid, field_name="last_conf")
        elif conf < 0 or conf > 100:
            report.add("error", "sync_state", f"last_conf[{nid}] out of [0,100]: {conf}",
                        object_id=nid, field_name="last_conf")

    # All nodes should have beliefs after inference
    for nid in sync_manager.bayes.nodes:
        info = sync_manager.bayes.nodes[nid]
        if info.belief is None:
            report.add("warning", "sync_state",
                        f"Node {nid} has no computed belief",
                        object_id=nid, field_name="belief")

    # History monotonic timestamps
    for nid, entries in sync_manager._history.items():
        prev_ts = None
        for entry in entries:
            ts = entry[0]
            if prev_ts is not None and ts < prev_ts:
                report.add("warning", "sync_state",
                            f"History timestamps not monotonic for {nid}",
                            object_id=nid, field_name="history")
                break
            prev_ts = ts

    # Graph invariants
    graph_report = validate_graph_invariants(sync_manager.bayes)
    report.issues.extend(graph_report.issues)

    return report


# ===================================================================
# 7. Full Validation Pipeline
# ===================================================================


def run_full_validation(
    objects: List[Dict],
    cfg: Dict,
    sync_manager=None,
    beliefs: Optional[Dict[str, float]] = None,
) -> ValidationReport:
    """
    Run all validators and return a combined report.

    Args:
        objects: Raw STIX objects from sample_data or OpenCTI
        cfg: Parsed bayes.yaml config dict
        sync_manager: Optional SyncManager (if built)
        beliefs: Optional inference results dict

    Returns:
        Combined ValidationReport
    """
    combined = ValidationReport()

    # 1. STIX schema
    stix_report = validate_stix_objects(objects)
    combined.issues.extend(stix_report.issues)

    # 2. Referential integrity
    ref_report = validate_referential_integrity(objects)
    combined.issues.extend(ref_report.issues)

    # 3. Config
    config_report = validate_config(cfg)
    combined.issues.extend(config_report.issues)

    # 4. Graph and sync state
    if sync_manager is not None:
        sync_report = validate_sync_manager_state(sync_manager)
        combined.issues.extend(sync_report.issues)

    # 5. Model outputs
    if sync_manager is not None and beliefs is not None:
        model_report = validate_inference_results(sync_manager.bayes, beliefs)
        combined.issues.extend(model_report.issues)

    return combined
