from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from datetime import datetime
from typing import Dict, List, Tuple

from stix2 import parse

import yaml
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "sample_data.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.calibration import summarize_calibration, validate_calibration_inputs, validate_bounds
from service.sync_manager import SyncManager
from service.eventbus import EventBus


def load_objects(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("objects", [])


def validate_stix(objects: List[Dict]) -> List[Tuple[int, str, str, str]]:
    errors = []
    for idx, obj in enumerate(objects):
        try:
            parse(obj, allow_custom=True)
        except Exception as exc:
            errors.append((idx, obj.get("type", ""), obj.get("id", ""), str(exc)))
    return errors


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def validate_referential_integrity(objects: List[Dict]) -> Tuple[int, List[str]]:
    ids = {o.get("id") for o in objects if o.get("id")}
    missing = []
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        src = obj.get("source_ref")
        dst = obj.get("target_ref")
        if src and src not in ids:
            missing.append(src)
        if dst and dst not in ids:
            missing.append(dst)
    return len(missing), sorted(set(missing))


def validate_timestamps(objects: List[Dict]) -> Tuple[int, int]:
    parse_errors = 0
    ordering_issues = 0
    for obj in objects:
        created = obj.get("created")
        modified = obj.get("modified")
        updated = obj.get("updated_at")
        parsed = {k: parse_timestamp(v) for k, v in (("created", created), ("modified", modified), ("updated_at", updated)) if v}
        if any(v is None for v in parsed.values()):
            parse_errors += 1
            continue
        if "created" in parsed and "modified" in parsed:
            if parsed["created"] > parsed["modified"]:
                ordering_issues += 1
        if "modified" in parsed and "updated_at" in parsed:
            if parsed["modified"] > parsed["updated_at"]:
                ordering_issues += 1
    return parse_errors, ordering_issues


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


def build_graph(objects: List[Dict], rels: List[Dict]):
    cfg_path = ROOT / "config" / "bayes.yaml"
    cfg = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
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
    objects = load_objects(DATA_PATH)
    rels = [o for o in objects if o.get("type") == "relationship"]
    parse_errors = validate_stix(objects)
    missing_ref_count, missing_refs = validate_referential_integrity(objects)
    ts_parse_errors, ts_order_issues = validate_timestamps(objects)
    values, missing, non_numeric, out_of_range = validate_confidence(objects)
    preds, outs = build_proxy_labels(objects)

    inputs_ok, input_issues = validate_calibration_inputs(preds, outs)
    summary = summarize_calibration(preds, outs, n_bins=5)

    non_rel_objects = [o for o in objects if o.get("type") != "relationship"]
    manager = build_graph(non_rel_objects, rels)
    posteriors = [info.belief if info.belief is not None else info.prior for info in manager.bayes.nodes.values()]
    belief_ok, belief_issues = validate_bounds(posteriors)
    max_in_degree = max((manager.bayes.G.in_degree(n) for n in manager.bayes.G.nodes), default=0)
    self_loops = list(nx.selfloop_edges(manager.bayes.G))
    edge_out_of_range = [
        (src, dst, w) for (src, dst), w in manager.bayes.edge_w.items() if w < 0.0 or w > 1.0
    ]

    print("Data validation (sample_data.json)")
    print(f"- STIX objects: {len(objects)}")
    print(f"- STIX parse errors: {len(parse_errors)}")
    print(f"- Missing relationship references: {missing_ref_count}")
    print(f"- Timestamp parse errors: {ts_parse_errors}")
    print(f"- Timestamp ordering issues: {ts_order_issues}")
    print(f"- Confidence present: {len(values)}")
    print(f"- Confidence missing: {len(missing)}")
    print(f"- Confidence non-numeric: {len(non_numeric)}")
    print(f"- Confidence out of range: {len(out_of_range)}")
    print(f"- Mean confidence (0-100): {round(mean(values), 1) if values else 'n/a'}")
    print("")
    print("Result validation (calibration, proxy labels from sightings)")
    print(f"- Inputs valid: {inputs_ok}")
    if not inputs_ok:
        print(f"- Input issues: {input_issues}")
    print(f"- Eligible objects (non-relationship with confidence): {summary.n_samples}")
    print(f"- Positive rate (proxy): {round(summary.positive_rate, 4)}")
    print(f"- Mean confidence: {round(summary.mean_confidence, 4)}")
    print(f"- Brier score: {round(summary.brier_score, 4)}")
    print(f"- ECE: {round(summary.ece, 4)}")
    print(f"- MCE: {round(summary.mce, 4)}")
    print(f"- n_bins: {summary.n_bins}")
    print("")
    print("Result validation (graph and posterior checks)")
    print(f"- Beliefs within [0,1]: {belief_ok}")
    if not belief_ok:
        print(f"- Belief issues: {belief_issues[:5]}")
    print(f"- Max in-degree: {max_in_degree}")
    print(f"- Self-loops: {len(self_loops)}")
    print(f"- Edge weights out of range: {len(edge_out_of_range)}")


if __name__ == "__main__":
    main()
