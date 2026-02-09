"""
Full-scale sample data validation script.

Validates sample_data.json using all available validators:
  - STIX schema validation
  - Referential integrity
  - Confidence field validation
  - Timestamp validation
  - Graph invariant checks
  - Model output validation (Noisy-OR monotonicity, convergence)
  - Calibration analysis (Brier, ECE, MCE)
  - Configuration validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import yaml
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "sample_data.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.calibration import (
    summarize_calibration,
    validate_calibration_inputs,
    validate_bounds,
)
from validation.data_validators import (
    validate_stix_objects,
    validate_referential_integrity,
    validate_graph_invariants,
    validate_config,
    validate_inference_results,
    validate_sync_manager_state,
    run_full_validation,
)
from service.sync_manager import SyncManager
from service.eventbus import EventBus


def load_objects(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("objects", [])


def validate_confidence(objects: List[Dict]) -> Tuple[List[float], List[Tuple[str, str]], List[Tuple[str, str, object]], List[Tuple[str, str, float]]]:
    values = []
    missing = []
    non_numeric = []
    out_of_range = []

    for obj in objects:
        if "confidence" not in obj or obj.get("confidence") is None:
            missing.append((obj.get("type", ""), obj.get("id", "")))
            continue
        value = obj.get("confidence")
        try:
            val_num = float(value)
        except Exception:
            non_numeric.append((obj.get("type", ""), obj.get("id", ""), value))
            continue
        if val_num < 0 or val_num > 100:
            out_of_range.append((obj.get("type", ""), obj.get("id", ""), val_num))
            continue
        values.append(val_num)

    return values, missing, non_numeric, out_of_range


def build_graph(objects: List[Dict], rels: List[Dict], cfg: Dict):
    manager = SyncManager(max_parents=5, bus=EventBus(), cfg=cfg)
    manager.build_from_opencti(objects, rels)
    return manager


def build_proxy_labels(objects: List[Dict]) -> Tuple[List[float], List[int]]:
    sighting_obs = set()
    for obj in objects:
        if obj.get("type") == "sighting":
            if obj.get("sighting_of_ref"):
                sighting_obs.add(obj["sighting_of_ref"])
            sighting_obs.update(obj.get("where_sighted_refs", []) or [])

    preds: List[float] = []
    outs: List[int] = []
    for obj in objects:
        if obj.get("type") == "relationship":
            continue
        conf = obj.get("confidence")
        if conf is None:
            continue
        preds.append(float(conf) / 100.0)
        outs.append(1 if obj.get("id") in sighting_obs else 0)

    return preds, outs


def main() -> None:
    # Load data and config
    objects = load_objects(DATA_PATH)
    cfg_path = ROOT / "config" / "bayes.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    rels = [o for o in objects if o.get("type") == "relationship"]
    non_rels = [o for o in objects if o.get("type") not in ("relationship", "sighting")]

    # ---- 1. STIX Schema Validation ----
    stix_report = validate_stix_objects(objects)

    # ---- 2. Referential Integrity ----
    ref_report = validate_referential_integrity(objects)

    # ---- 3. Config Validation ----
    config_report = validate_config(cfg)

    # ---- 4. Confidence Field Validation ----
    values, missing, non_numeric, out_of_range = validate_confidence(objects)

    # ---- 5. Build Graph and Run Inference ----
    manager = build_graph(non_rels, rels, cfg)
    beliefs = manager.bayes.infer_all()
    manager.run_inference_and_diff()

    # ---- 6. Graph Invariant Validation ----
    graph_report = validate_graph_invariants(manager.bayes)

    # ---- 7. Model Output Validation ----
    model_report = validate_inference_results(manager.bayes, beliefs)

    # ---- 8. SyncManager State Validation ----
    sync_report = validate_sync_manager_state(manager)

    # ---- 9. Calibration Analysis ----
    preds, outs = build_proxy_labels(objects)
    inputs_ok, input_issues = validate_calibration_inputs(preds, outs)
    summary = summarize_calibration(preds, outs, n_bins=5)

    # ---- 10. Full Pipeline Validation ----
    full_report = run_full_validation(objects, cfg, sync_manager=manager, beliefs=beliefs)

    # ---- 11. Posterior bounds check ----
    posteriors = [info.belief if info.belief is not None else info.prior
                  for info in manager.bayes.nodes.values()]
    belief_ok, belief_issues = validate_bounds(posteriors)
    max_in_degree = max((manager.bayes.G.in_degree(n) for n in manager.bayes.G.nodes), default=0)
    self_loops = list(nx.selfloop_edges(manager.bayes.G))
    edge_out_of_range = [
        (src, dst, w) for (src, dst), w in manager.bayes.edge_w.items() if w < 0.0 or w > 1.0
    ]

    # ---- 12. Convergence Info ----
    conv_info = manager.bayes.get_convergence_info()

    # ================================================================
    # Output
    # ================================================================

    print("=" * 64)
    print("  FULL-SCALE DATA VALIDATION REPORT")
    print("=" * 64)
    print()

    print("1. STIX Schema Validation")
    print(f"   Objects: {len(objects)}")
    print(f"   Errors: {stix_report.error_count}")
    print(f"   Warnings: {stix_report.warning_count}")
    if stix_report.errors:
        for e in stix_report.errors[:5]:
            print(f"     {e}")
    print()

    print("2. Referential Integrity")
    print(f"   Errors: {ref_report.error_count}")
    print(f"   Warnings: {ref_report.warning_count}")
    if ref_report.warnings:
        for w in ref_report.warnings[:5]:
            print(f"     {w}")
    print()

    print("3. Configuration Validation")
    print(f"   Valid: {config_report.is_valid}")
    print(f"   Errors: {config_report.error_count}")
    print(f"   Warnings: {config_report.warning_count}")
    if config_report.issues:
        for i in config_report.issues[:5]:
            print(f"     {i}")
    print()

    print("4. Confidence Field Validation")
    print(f"   Present: {len(values)}")
    print(f"   Missing: {len(missing)}")
    print(f"   Non-numeric: {len(non_numeric)}")
    print(f"   Out of range: {len(out_of_range)}")
    print(f"   Mean confidence (0-100): {round(mean(values), 1) if values else 'n/a'}")
    print()

    print("5. Graph Structure")
    print(f"   Nodes: {len(manager.bayes.nodes)}")
    print(f"   Edges: {len(list(manager.bayes.G.edges()))}")
    print(f"   Max in-degree: {max_in_degree}")
    print(f"   Self-loops: {len(self_loops)}")
    print(f"   Edge weights out of range: {len(edge_out_of_range)}")
    print(f"   Beliefs within [0,1]: {belief_ok}")
    print(f"   Graph invariant errors: {graph_report.error_count}")
    print(f"   Graph invariant warnings: {graph_report.warning_count}")
    print()

    print("6. Model Output Validation")
    print(f"   Errors: {model_report.error_count}")
    print(f"   Warnings: {model_report.warning_count}")
    if model_report.issues:
        for i in model_report.issues[:5]:
            print(f"     {i}")
    print()

    print("7. Convergence Info")
    if conv_info:
        for ci in conv_info:
            print(f"   SCC size={ci['scc_size']}, iters={ci['iterations']}, converged={ci['converged']}")
    else:
        print("   No cyclic components (pure DAG)")
    print()

    print("8. Calibration Analysis (proxy labels from sightings)")
    print(f"   Inputs valid: {inputs_ok}")
    if not inputs_ok:
        print(f"   Issues: {input_issues}")
    print(f"   Eligible objects: {summary.n_samples}")
    print(f"   Positive rate (proxy): {round(summary.positive_rate, 4)}")
    print(f"   Mean confidence: {round(summary.mean_confidence, 4)}")
    print(f"   Brier score: {round(summary.brier_score, 4)}")
    print(f"   ECE: {round(summary.ece, 4)}")
    print(f"   MCE: {round(summary.mce, 4)}")
    print(f"   Bins: {summary.n_bins}")
    print()

    print("9. Full Validation Pipeline Summary")
    print(f"   Total issues: {len(full_report.issues)}")
    print(f"   Errors: {full_report.error_count}")
    print(f"   Warnings: {full_report.warning_count}")
    print(f"   VALID: {full_report.is_valid}")
    print(f"   By category: {full_report.summary()['by_category']}")
    print()

    print("=" * 64)
    if full_report.is_valid:
        print("  RESULT: ALL VALIDATION CHECKS PASSED")
    else:
        print("  RESULT: VALIDATION ERRORS FOUND")
        for e in full_report.errors:
            print(f"    {e}")
    print("=" * 64)

    sys.exit(0 if full_report.is_valid else 1)


if __name__ == "__main__":
    main()
