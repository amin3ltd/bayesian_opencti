from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from stix2 import parse

from validation.calibration import summarize_calibration, validate_calibration_inputs


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "sample_data.json"


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
    parse_errors = validate_stix(objects)
    values, missing, non_numeric, out_of_range = validate_confidence(objects)
    preds, outs = build_proxy_labels(objects)

    inputs_ok, input_issues = validate_calibration_inputs(preds, outs)
    summary = summarize_calibration(preds, outs, n_bins=5)

    print("Data validation (sample_data.json)")
    print(f"- STIX objects: {len(objects)}")
    print(f"- STIX parse errors: {len(parse_errors)}")
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


if __name__ == "__main__":
    main()
