# Bayesian Confidence Scoring for OpenCTI

Academic research project for Bayesian confidence propagation over threat
intelligence graphs. The system combines Noisy-OR inference with damped
fixed-point updates for cyclic graphs to estimate posterior confidence while
preserving interpretability.

## How it works

1. Ingest STIX objects and relationships from OpenCTI (or `sample_data.json`
   fallback when OpenCTI is not available).
2. Normalize entity types, apply default priors, and apply time decay using
   `updated_at`/`modified`/`created` timestamps.
3. Build a directed graph with weighted edges, enforcing a maximum parent cap
   for stability.
4. Run inference:
   - DAG components: exact Noisy-OR update.
   - Cyclic SCCs: damped fixed-point iteration to convergence.
5. Smooth outputs (EMA) and apply delta thresholds before pushing updates to
   OpenCTI.
6. Expose results via Flask API and Server-Sent Events (SSE) for the dashboard.

Noisy-OR update:

```
P(n) = 1 - (1 - prior_n) * Π_{p in parents} (1 - w_pn * P(p))
```

Time decay:

```
prior' = prior * 0.5^(age_days / half_life)
```

## Architecture and components

| Component | Responsibility |
| --- | --- |
| `run.py` | Entry point (starts MainApp). |
| `service/app.py` | Application bootstrap, OpenCTI fetch, polling, Flask setup. |
| `service/opencti_client.py` | OpenCTI API client (fetch and update confidence). |
| `service/sync_manager.py` | Graph build, priors/weights/decay, inference diffs. |
| `bayes/model.py` | Noisy-OR inference engine and SCC solver. |
| `service/eventbus.py` | Pub/sub bus for SSE updates. |
| `api/server.py` | Flask API + SSE endpoints. |
| `dashboard/` | UI (Cytoscape visualization, controls, history). |
| `validation/calibration.py` | Data validation and calibration metrics. |
| `scripts/validate_sample_data.py` | Reproducible data + result validation. |

## Installation (all platforms)

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
```

### Windows (PowerShell)

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

### Windows (cmd.exe)

```bat
py -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
python run.py
```

### Docker

```bash
cd docker
cp .env.sample .env
docker-compose up -d
```

### Conda (optional)

```bash
conda create -n bayesian-opencti python=3.11
conda activate bayesian-opencti
pip install -r requirements.txt
cp .env.example .env
python run.py
```

### Access

- Dashboard: http://localhost:5000
- API: http://localhost:5000/api
- Health check: http://localhost:5000/api/v1/status

## Configuration

Environment variables in `.env`:
- `OPENCTI_URL`, `OPENCTI_TOKEN`: OpenCTI connection.
- `LOG_LEVEL`: logging verbosity (default: INFO).
- `POLL_INTERVAL_SECONDS`: polling interval for updates.
- `MAX_PARENTS_PER_NODE`: cap for incoming edges per node.

YAML parameters in `config/bayes.yaml`:
- `lbp_damping`, `lbp_epsilon`, `lbp_max_iters`: fixed-point solver.
- `rel_type_weight`, `default_rel_weight`: relationship weighting.
- `time_decay_half_life`: per-type decay in days.
- `ema_alpha`: output smoothing.
- `confidence_push_delta_min`: push threshold (percentage points).

## API endpoints

- `GET /api/v1/status`: health status.
- `GET /api/v1/network`: nodes + edges for visualization.
- `POST /api/v1/recompute`: recompute and (optionally) push to OpenCTI.
- `GET /api/v1/node?id=...`: node details.
- `GET /api/v1/contributions?id=...`: parent contribution breakdown.
- `GET /api/v1/paths?id=...`: top evidence paths.
- `GET /api/v1/history?id=...`: confidence history.
- `GET /api/v1/config`: current Bayesian configuration.
- `GET /api/v1/stream`: SSE event stream.

## UI (dashboard)

- Search by label or ID
- Clear search and restore view
- Fit/Refresh/Recompute actions
- Graph export to JSON
- Theme toggle (light/dark)
- Last update timestamp
- Parent contributions, evidence paths, and history chart

## Testing and validation (full procedure)

### 1) Unit tests

Fast, isolated tests for inference and validation utilities.

```bash
python3 -m pytest -m unit
```

### 2) Integration tests

Tests that combine multiple modules (SyncManager + API surfaces).

```bash
python3 -m pytest -m integration
```

### 3) End-to-end (E2E) tests

End-to-end API flows using a real SyncManager and the Flask test client.

```bash
python3 -m pytest -m e2e
```

### 4) Full suite (recommended)

```bash
python3 -m pytest
```

Result (latest): **78 passed**

### 5) Data validation procedure

```bash
python3 scripts/validate_sample_data.py
```

What it checks (data):
- STIX 2.1 parse validation for each object.
- Required fields and referential integrity for relationships.
- Timestamp parsing and ordering checks (`created` ≤ `modified` ≤ `updated_at`).
- Confidence bounds: numeric and within [0, 100].

### 6) Result validation procedure (calibration)

The same script performs calibration using proxy labels derived from sightings:
`outcome = 1` if an entity is referenced by `sighting_of_ref` or
`where_sighted_refs`, else `0`. The workflow is:

1. Collect confidence predictions from non-relationship objects.
2. Build proxy outcomes from sightings.
3. Validate inputs and compute summary metrics (Brier, ECE, MCE).

This is a sparse proxy signal and should be interpreted as a sanity check rather
than ground-truth calibration.

### 7) Result validation procedure (graph/posterior checks)

The same script performs model-level checks after building the graph:
- Posterior bounds: all beliefs must be within [0, 1].
- Graph integrity: no self-loops, weights within [0, 1].
- Parent cap: max in-degree should respect the configured limit.

## Validation outcomes (sample_data.json)

Data validation results:
- STIX objects: 47
- STIX parse errors: 0
- Missing relationship references: 0
- Timestamp parse errors: 0
- Timestamp ordering issues: 0
- Confidence present: 37
- Confidence missing: 10
- Confidence non-numeric: 0
- Confidence out of range: 0
- Mean confidence (0-100): 77.7

Result validation (calibration, proxy labels):
- Eligible objects (non-relationship with confidence): 11
- Positive rate (proxy): 0.0909
- Mean confidence: 0.7227
- Brier score: 0.6202
- ECE: 0.7773
- MCE: 0.8667
- n_bins: 5

Graph/posterior validation:
- Beliefs within [0, 1]: True
- Max in-degree: 5
- Self-loops: 0
- Edge weights out of range: 0

## Additional validation techniques (recommended)

These techniques are not fully automated in the current scripts but are useful
for research-grade validation:

- **Reliability diagrams**: compare predicted confidence vs. observed accuracy.
- **Sensitivity analysis**: perturb priors/weights to measure output stability.
- **Temporal stability**: detect oscillations or drift in confidence history.
- **Cross-source agreement**: compare with external feeds (MISP, VT, OTX).
- **Ablation studies**: measure impact of removing edge types or time decay.
- **Holdout evaluation**: use labeled outcomes when available for calibration.

## Assumptions and limitations

- Proxy labels from sightings are **not** ground truth; calibration is indicative.
- Confidence propagation depends on relationship coverage and quality.
- OpenCTI credentials and network access are required for live data ingestion.
- API endpoints are unauthenticated; deploy behind trusted network controls.

## License

MIT License
