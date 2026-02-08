# Bayesian Confidence Scoring for OpenCTI

Academic research project for Bayesian confidence propagation over threat
intelligence graphs. The system combines Noisy-OR inference with damped
fixed-point updates for cyclic graphs to estimate posterior confidence while
preserving interpretability.

## How it works (brief)

1) Ingest STIX objects and relationships from OpenCTI (or `sample_data.json`).
2) Build a directed graph with weighted edges (relationship confidence and type weights).
3) Run Noisy-OR inference for DAG components and damped fixed-point for SCCs.
4) Apply smoothing and delta thresholds before pushing updates to OpenCTI.
5) Serve results via Flask API and SSE for live dashboard updates.

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
```

Dashboard: http://localhost:5000

## Configuration

Primary settings:
- `.env`: OpenCTI URL/token, poll interval, max parents.
- `config/bayes.yaml`: `lbp_damping`, `lbp_epsilon`, `lbp_max_iters`,
  `rel_type_weight`, `time_decay_half_life`, `ema_alpha`,
  `confidence_push_delta_min`.

## Validation (outcomes)

### Data validation (sample_data.json)

Computed using STIX 2.1 parsing and confidence bounds checks:
- STIX objects: 47
- STIX parse errors: 0
- Confidence present: 37
- Confidence missing: 10
- Confidence non-numeric: 0
- Confidence out of range: 0
- Mean confidence (0-100): 77.7

### Result validation (calibration)

Computed with `validation.calibration.summarize_calibration` using sightings as
proxy labels (outcome = 1 if referenced by `sighting_of_ref` or
`where_sighted_refs`, else 0). This is a sparse proxy and should be interpreted
as a sanity check rather than a definitive calibration study.

- Eligible objects (non-relationship with confidence): 11
- Positive rate (proxy): 0.0909
- Mean confidence: 0.7227
- Brier score: 0.6202
- ECE: 0.7773
- MCE: 0.8667
- n_bins: 5

## Test results

```
python3 -m pytest
```

Result: 77 passed

## License

MIT License
