# Bayesian Confidence Scoring for OpenCTI

Academic research project for Bayesian confidence propagation over threat
intelligence graphs. The system combines Noisy-OR inference with damped
fixed-point updates for cyclic graphs to estimate posterior confidence while
preserving interpretability.

## Requirements

- **Python**: 3.8, 3.9, 3.10, or 3.11 (tested on all versions)
- **OpenCTI**: 6.7.5+ (optional, falls back to sample data if unavailable)
- **Dependencies**: See `requirements.txt` for full list

### Key Dependencies

- `networkx>=3.2` - Graph structure and SCC analysis
- `numpy>=1.26` - Numerical computations
- `pandas>=2.2` - Data manipulation
- `pycti>=6.8.0` - OpenCTI API client
- `Flask>=3.0` - Web API and dashboard server
- `pydantic>=2.11` - Settings and configuration validation
- `scikit-learn>=1.3` - Calibration metrics
- `scipy>=1.11` - Scientific computing utilities

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
| `service/eventbus.py` | Pub/sub bus for SSE updates. |
| `service/logging_setup.py` | Centralized logging configuration. |
| `bayes/model.py` | Noisy-OR inference engine and SCC solver. |
| `api/server.py` | Flask API + SSE endpoints. |
| `dashboard/` | UI (Cytoscape visualization, controls, history). |
| `config/settings.py` | Pydantic-based settings management. |
| `config/bayes.yaml` | Bayesian model configuration (weights, decay, etc.). |
| `validation/calibration.py` | Data validation and calibration metrics. |
| `validation/data_validators.py` | STIX data validation utilities. |
| `scripts/validate_sample_data.py` | Reproducible data + result validation. |
| `scripts/mock_data_loader.py` | Mock data loading utilities. |
| `tests/` | Comprehensive test suite (unit, integration, e2e). |
| `docker/` | Docker Compose setup for OpenCTI platform (optional). |

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

### Docker (OpenCTI Platform)

The `docker/` directory contains Docker Compose configuration for running the full OpenCTI platform (including Redis, Elasticsearch, MinIO, RabbitMQ, and OpenCTI services). This is optional and only needed if you want to run OpenCTI locally.

```bash
cd docker
cp .env.sample .env
# Edit .env with your configuration
docker-compose up -d
```

**Note**: The Bayesian Confidence Service itself runs as a Python application (see other installation methods above). The Docker setup is only for the OpenCTI platform infrastructure.

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

### Environment Variables (`.env`)

Copy `.env.example` to `.env` and configure:

- `OPENCTI_URL`: OpenCTI API URL (default: `http://127.0.0.1:8080`)
- `OPENCTI_TOKEN`: OpenCTI API token (required for live data)
- `LOG_LEVEL`: Logging verbosity - DEBUG, INFO, WARNING, ERROR (default: `INFO`)
- `POLL_INTERVAL_SECONDS`: Polling interval for OpenCTI updates in seconds (default: `30`)
- `MAX_PARENTS_PER_NODE`: Maximum incoming edges per node for stability (default: `5`)

### Bayesian Model Configuration (`config/bayes.yaml`)

- **Fixed-point solver**:
  - `lbp_damping`: Damping factor for cyclic SCC convergence (default: `0.55`)
  - `lbp_epsilon`: Convergence threshold (default: `1.0e-4`)
  - `lbp_max_iters`: Maximum iterations for fixed-point solver (default: `100`)

- **Relationship weighting**:
  - `default_rel_weight`: Default edge weight when type not specified (default: `0.50`)
  - `rel_type_weight`: Per-relationship-type weights (e.g., `indicates: 0.85`, `uses: 0.35`)
  - `rel_conf_fallback`: Fallback confidence when relationship confidence is 0/None (default: `50`)
  - `report_object_min`: Minimum weight for report→object edges (default: `30`)

- **Time decay**:
  - `time_decay_half_life`: Half-life in days per entity type (e.g., `Indicator: 60`, `Malware: 180`)

- **Output stability**:
  - `ema_alpha`: Exponential moving average smoothing factor 0..1 (default: `0.35`)
  - `confidence_push_delta_min`: Minimum percentage point change to push update (default: `2`)

## API endpoints

All endpoints return JSON unless otherwise specified.

- `GET /api/v1/status` - Health check endpoint
- `GET /api/v1/network` - Export full graph (nodes + edges) for visualization
- `POST /api/v1/recompute` - Trigger recomputation and optionally push updates to OpenCTI
- `GET /api/v1/node?id=<node_id>` - Get node details (prior, posterior, type, name)
- `GET /api/v1/contributions?id=<node_id>&topk=<N>` - Parent contribution breakdown (default topk=10)
- `GET /api/v1/paths?id=<node_id>&k=<N>&maxlen=<M>` - Top evidence paths (default k=5, maxlen=3)
- `GET /api/v1/history?id=<node_id>` - Confidence history for a node
- `GET /api/v1/config` - Current Bayesian model configuration from `config/bayes.yaml`
- `GET /api/v1/stream` - Server-Sent Events (SSE) stream for real-time updates

## UI (dashboard)

The web dashboard provides an interactive visualization of the threat intelligence graph with Bayesian confidence scores.

### Features

- **Interactive graph visualization** using Cytoscape.js
- **Search functionality** - Search nodes by label or ID
- **Node details panel** - View prior, posterior, type, and metadata
- **Contribution analysis** - See which parent nodes contribute most to a node's confidence
- **Evidence paths** - Explore top evidence paths leading to a node
- **Confidence history** - View confidence changes over time (chart)
- **Graph controls**:
  - Fit view to graph
  - Refresh data
  - Trigger recomputation
  - Export graph to JSON
- **Theme toggle** - Switch between light and dark themes
- **Real-time updates** - Server-Sent Events (SSE) for live confidence updates
- **Last update timestamp** - Shows when data was last refreshed

Access the dashboard at `http://localhost:5000` after starting the service.

## Development

### Code Quality

The project uses several code quality tools configured in `pyproject.toml`:

- **Black** - Code formatting (line length: 100)
- **flake8** - Linting (configured in `.flake8`)
- **mypy** - Type checking (Python 3.8+)

Run formatting:
```bash
black .
```

Run linting:
```bash
flake8 .
```

Run type checking:
```bash
mypy .
```

### Testing and validation (full procedure)

#### 1) Unit tests

Fast, isolated tests for inference and validation utilities.

```bash
python3 -m pytest -m unit
```

#### 2) Integration tests

Tests that combine multiple modules (SyncManager + API surfaces).

```bash
python3 -m pytest -m integration
```

#### 3) End-to-end (E2E) tests

End-to-end API flows using a real SyncManager and the Flask test client.

```bash
python3 -m pytest -m e2e
```

#### 4) Full test suite (recommended)

```bash
python3 -m pytest
```

With coverage:
```bash
python3 -m pytest --cov=. --cov-report=html --cov-report=term-missing
```

**Latest test results**: 78 tests passed

#### 5) CI/CD

The project includes GitHub Actions CI/CD (`.github/workflows/ci.yml`) that runs:
- Linting (flake8, black, mypy)
- Tests across Python 3.8, 3.9, 3.10, 3.11
- Validation tests
- Security checks (safety, bandit)

### 5) Data and Result Validation

The project includes comprehensive validation tools to ensure data quality and model correctness. Run the full validation script:

```bash
python3 scripts/validate_sample_data.py
```

This script performs **12 validation checks** and outputs a detailed report. Here's what each check validates:

#### A. Data Validation

**1. STIX Schema Validation**
- Validates STIX 2.1 object structure
- Checks required fields (`id`, `type`)
- Validates STIX ID format (`type--uuid`)
- Verifies ID prefix matches declared type
- Checks for duplicate IDs
- Validates known STIX types
- **What to look for**: Errors indicate malformed data that may cause ingestion failures

**2. Referential Integrity**
- Validates that all `source_ref` and `target_ref` in relationships point to existing objects
- Checks `object_refs` in reports
- Validates `sighting_of_ref` and `where_sighted_refs` in sightings
- **What to look for**: Warnings indicate broken references that may create disconnected graph components

**3. Configuration Validation**
- Validates `config/bayes.yaml` parameters
- Checks ranges: `lbp_damping` (0.01-0.99), `lbp_epsilon` (>0), `lbp_max_iters` (>0)
- Validates `ema_alpha` [0, 1], `confidence_push_delta_min` [0, 100]
- Checks `rel_type_weight` values are in [0, 1]
- Validates `time_decay_half_life` values are positive
- **What to look for**: Errors indicate invalid configuration that may cause inference failures

**4. Confidence Field Validation**
- Checks confidence values are numeric
- Validates confidence is within [0, 100]
- Detects NaN/Inf values
- Reports missing confidence fields
- Computes mean confidence statistics
- **What to look for**: Errors indicate data quality issues; missing confidence will use default priors

**5. Timestamp Validation**
- Parses ISO 8601 timestamps (`created`, `modified`, `updated_at`)
- Validates timestamp ordering: `created` ≤ `modified`
- **What to look for**: Errors indicate invalid timestamps; ordering issues may affect time decay calculations

#### B. Graph and Model Validation

**6. Graph Structure Validation**
- Counts nodes and edges
- Checks for self-loops (should be 0)
- Validates edge weights are in [0, 1] and not NaN/Inf
- Verifies parent cap (`MAX_PARENTS_PER_NODE`) is respected
- Checks graph is not empty
- **What to look for**: Errors indicate graph construction problems; warnings about parent cap violations suggest need for tuning

**7. Model Output Validation**
- Validates all beliefs are computed (no missing values)
- Checks beliefs are within [0, 1] and not NaN/Inf
- Verifies Noisy-OR monotonicity: for acyclic nodes with parents, posterior ≥ prior
- **What to look for**: Errors indicate inference failures; monotonicity warnings suggest potential model issues

**8. Convergence Information**
- Reports convergence statistics for cyclic SCCs (Strongly Connected Components)
- Shows SCC size, iterations used, and convergence status
- **What to look for**: Non-converged SCCs may indicate need to increase `lbp_max_iters` or adjust `lbp_damping`

**9. SyncManager State Validation**
- Validates `last_conf` values are in [0, 100]
- Checks all nodes have computed beliefs after inference
- Validates history timestamps are monotonic
- **What to look for**: Errors indicate state corruption; warnings suggest incomplete inference

**10. Posterior Bounds Check**
- Verifies all posterior beliefs are within [0, 1]
- Checks for NaN/Inf values in beliefs
- **What to look for**: Errors indicate numerical instability or model bugs

#### C. Calibration Validation

**11. Calibration Analysis (Proxy Labels)**
- Builds proxy binary labels from sightings: `outcome = 1` if entity is referenced by `sighting_of_ref` or `where_sighted_refs`, else `0`
- Computes calibration metrics:
  - **Brier Score**: Mean squared error between predictions and outcomes (lower is better, 0 = perfect)
  - **ECE (Expected Calibration Error)**: Weighted average of calibration error across bins
  - **MCE (Maximum Calibration Error)**: Maximum calibration error in any bin
- **What to look for**: 
  - High Brier score (>0.5) suggests poor calibration
  - High ECE/MCE (>0.3) indicates systematic over/under-confidence
  - **Note**: Proxy labels from sightings are sparse and not ground truth; use as sanity check only

**12. Full Pipeline Validation**
- Runs all validators and aggregates results
- Provides summary by category
- **What to look for**: Overall `VALID: True/False` indicates if all critical checks passed

#### Interpreting Validation Results

**Success Criteria:**
- ✅ All validation checks show `VALID: True`
- ✅ Zero errors in STIX schema, referential integrity, and model output
- ✅ Graph structure is valid (no self-loops, weights in range)
- ✅ All beliefs computed and within bounds
- ✅ Convergence achieved for all cyclic components

**Common Issues and Fixes:**

1. **STIX Schema Errors**: Fix malformed objects in source data
2. **Referential Integrity Warnings**: May indicate incomplete data bundle; acceptable if references exist in OpenCTI
3. **Graph Structure Errors**: Check data quality; self-loops should be filtered during ingestion
4. **Non-Convergence**: Increase `lbp_max_iters` or adjust `lbp_damping` in `config/bayes.yaml`
5. **Calibration Issues**: Proxy labels are sparse; consider collecting ground-truth labels for proper calibration

#### Programmatic Validation

You can also use validation functions programmatically:

```python
from validation.data_validators import (
    validate_stix_objects,
    validate_referential_integrity,
    validate_graph_invariants,
    validate_config,
    validate_inference_results,
    run_full_validation
)
from validation.calibration import (
    brier_score,
    expected_calibration_error,
    summarize_calibration
)

# Validate STIX objects
stix_report = validate_stix_objects(objects)
print(f"Errors: {stix_report.error_count}, Warnings: {stix_report.warning_count}")

# Validate configuration
config_report = validate_config(cfg)
if not config_report.is_valid:
    for error in config_report.errors:
        print(error)

# Compute calibration metrics
predictions = [0.7, 0.8, 0.9]  # confidence scores [0, 1]
outcomes = [1, 1, 0]  # binary outcomes
brier = brier_score(predictions, outcomes)
ece, mce = expected_calibration_error(predictions, outcomes, n_bins=5)
summary = summarize_calibration(predictions, outcomes, n_bins=5)
print(f"Brier: {summary.brier_score:.4f}, ECE: {summary.ece:.4f}, MCE: {summary.mce:.4f}")
```

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

## Advanced Validation Techniques (Step-by-Step)

These techniques are not fully automated in the current scripts but are useful
for research-grade validation. Each procedure below outlines how to execute it.

### 1) Reliability Diagrams (Calibration Curves)

**Purpose**: Visualize predicted confidence vs. observed accuracy to assess calibration quality.

**Steps**:
1. Collect a dataset of predictions `p_i` (confidence scores 0-1) and binary outcomes `y_i` (0 or 1).
2. Bin predictions into confidence buckets (e.g., 0-0.1, 0.1-0.2, ..., 0.9-1.0).
3. For each bin, compute:
   - Mean confidence (average of `p_i` in bin)
   - Empirical accuracy (average of `y_i` in bin)
   - Bin count (number of samples)
4. Plot mean confidence (x-axis) vs. accuracy (y-axis).
5. Add diagonal line (perfect calibration).
6. Measure gap to diagonal as calibration error (ECE/MCE).

**Implementation**:
```python
from validation.calibration import analyze_calibration
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Get predictions and outcomes (from your data)
predictions = [...]  # List of confidence scores [0, 1]
outcomes = [...]     # List of binary outcomes [0, 1]

# Compute calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    outcomes, predictions, n_bins=10, strategy='uniform'
)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
plt.xlabel("Mean Predicted Confidence")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()

# Get detailed metrics
result = analyze_calibration(predictions, outcomes, n_bins=10)
print(f"Brier Score: {result.brier_score:.4f}")
print(f"ECE: {result.ece:.4f}, MCE: {result.mce:.4f}")
```

**Interpretation**: Points above diagonal = overconfident, below = underconfident. Good calibration should follow the diagonal closely.

### 2) Sensitivity Analysis (Robustness)

**Purpose**: Evaluate model stability under parameter changes to identify sensitive parameters and nodes.

**Steps**:
1. Choose parameters to perturb:
   - Model parameters: `lbp_damping`, `lbp_epsilon`, `lbp_max_iters`
   - Relationship weights: `rel_type_weight`, `default_rel_weight`
   - Time decay: `time_decay_half_life` values
   - Node priors: default priors per entity type
2. Create baseline configuration (current `config/bayes.yaml`).
3. Create perturbed configurations (e.g., ±5%, ±10%, ±20%).
4. For each configuration:
   - Build graph from same data
   - Run inference
   - Store beliefs for all nodes
5. Compare outputs:
   - Compute per-node absolute deltas: `|belief_perturbed - belief_baseline|`
   - Identify max/mean/median delta across nodes
   - Flag nodes with large deltas (>0.1) as sensitive
6. Document which parameters cause largest changes.

**Implementation**:
```python
import yaml
from service.sync_manager import SyncManager
from service.eventbus import EventBus

# Load baseline config
with open('config/bayes.yaml') as f:
    baseline_cfg = yaml.safe_load(f)

# Create perturbed configs
perturbations = {
    'damping_high': {**baseline_cfg, 'lbp_damping': baseline_cfg['lbp_damping'] * 1.1},
    'damping_low': {**baseline_cfg, 'lbp_damping': baseline_cfg['lbp_damping'] * 0.9},
}

baseline_manager = SyncManager(max_parents=5, bus=EventBus(), cfg=baseline_cfg)
baseline_manager.build_from_opencti(objects, relationships)
baseline_beliefs = baseline_manager.bayes.infer_all()

sensitivity_results = {}
for name, cfg in perturbations.items():
    manager = SyncManager(max_parents=5, bus=EventBus(), cfg=cfg)
    manager.build_from_opencti(objects, relationships)
    beliefs = manager.bayes.infer_all()
    
    deltas = {nid: abs(beliefs[nid] - baseline_beliefs[nid]) 
              for nid in baseline_beliefs}
    sensitivity_results[name] = {
        'max_delta': max(deltas.values()),
        'mean_delta': sum(deltas.values()) / len(deltas),
        'sensitive_nodes': [nid for nid, d in deltas.items() if d > 0.1]
    }

print("Sensitivity Analysis Results:")
for name, result in sensitivity_results.items():
    print(f"{name}: max_delta={result['max_delta']:.4f}, "
          f"mean_delta={result['mean_delta']:.4f}, "
          f"sensitive_nodes={len(result['sensitive_nodes'])}")
```

**Interpretation**: High sensitivity (>0.1 delta) indicates need for careful parameter tuning. Low sensitivity (<0.05) suggests robust model.

### 3) Temporal Stability (Drift Detection)

**Purpose**: Ensure confidence scores do not oscillate unrealistically over time and detect drift.

**Steps**:
1. Collect confidence history:
   - Use API endpoint: `GET /api/v1/history?id=<node_id>`
   - Or access `sync_manager._history` directly
   - History format: `[(timestamp, old_conf, new_conf), ...]`
2. For each node, compute:
   - Per-update deltas: `|new_conf - old_conf|`
   - Rolling variance over a window (e.g., last 10 updates)
   - Maximum jump: largest absolute change
   - Reversal count: number of times confidence increases then decreases (or vice versa)
3. Flag anomalies:
   - Sudden jumps: delta > threshold (e.g., >20 percentage points)
   - High variance: rolling variance > threshold
   - Frequent reversals: >N reversals in window
4. Correlate with data changes:
   - Check if jumps coincide with relationship additions/deletions
   - Verify timestamps match OpenCTI update events
   - Check if time decay is causing expected gradual changes

**Implementation**:
```python
import requests
from validation.calibration import validate_monotonicity
from statistics import variance

# Get history via API
node_id = "indicator--abc123"
response = requests.get(f"http://localhost:5000/api/v1/history?id={node_id}")
history = response.json()['history']  # [(timestamp, old, new), ...]

# Compute metrics
timestamps = [h[0] for h in history]
values = [h[2] / 100.0 for h in history]  # Convert to [0, 1]
deltas = [abs(values[i] - values[i-1]) for i in range(1, len(values))]

# Check monotonicity
is_valid, issues = validate_monotonicity(timestamps, values, max_jump=0.2)
if not is_valid:
    print("Temporal stability issues detected:")
    for issue in issues:
        print(f"  {issue}")

# Compute rolling variance
window_size = 10
if len(values) >= window_size:
    rolling_var = variance(values[-window_size:])
    print(f"Rolling variance (last {window_size}): {rolling_var:.6f}")

# Detect reversals
reversals = 0
for i in range(2, len(values)):
    if (values[i-1] > values[i-2] and values[i] < values[i-1]) or \
       (values[i-1] < values[i-2] and values[i] > values[i-1]):
        reversals += 1

print(f"Reversals: {reversals}, Max delta: {max(deltas) if deltas else 0:.4f}")
```

**Interpretation**: 
- Expected: Gradual changes due to time decay, smooth updates from new relationships
- Anomalous: Sudden jumps (>20pp), frequent reversals, high variance suggest instability
- Action: If unstable, check EMA smoothing (`ema_alpha`), increase `confidence_push_delta_min`, or investigate data quality

### 4) Cross-Source Agreement

**Purpose**: Compare model confidence with external threat intelligence feeds to validate predictions.

**Steps**:
1. Map indicators/entities to external sources:
   - Indicators: Extract hashes (MD5, SHA1, SHA256), domains, IPs, URLs
   - Entities: Map to external IDs where available
2. Fetch external scores:
   - **VirusTotal**: Detection ratio (detections/total_scans) → normalize to [0, 1]
   - **MISP**: Threat level → map to confidence [0, 1]
   - **OTX (AlienVault)**: Pulse count or reputation → normalize
   - **Shodan**: Risk score → normalize
3. Normalize external scores to [0, 1] range.
4. Compute agreement metrics:
   - Correlation coefficient (Pearson/Spearman)
   - Mean absolute error (MAE)
   - Disagreement rate: % of cases where |model_conf - external_conf| > threshold
   - Confusion matrix: high model / low external vs. low model / high external
5. Investigate discrepancies:
   - High model confidence but low external → possible false positive
   - Low model confidence but high external → possible false negative
   - Check if discrepancies correlate with entity types or relationship patterns

**Implementation**:
```python
import requests
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Example: Compare with VirusTotal
def fetch_vt_score(hash_value, api_key):
    """Fetch VirusTotal detection ratio for a hash."""
    url = f"https://www.virustotal.com/api/v3/files/{hash_value}"
    headers = {"x-apikey": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        stats = data['data']['attributes']['last_analysis_stats']
        total = stats['harmless'] + stats['malicious'] + stats['suspicious']
        detections = stats['malicious'] + stats['suspicious']
        return detections / total if total > 0 else 0.0
    return None

# Collect model predictions and external scores
model_confidences = []  # From your model [0, 1]
external_scores = []    # From external sources [0, 1]
indicators = [...]      # List of indicator objects

for indicator in indicators:
    # Get model confidence
    node_id = indicator['id']
    model_conf = get_model_confidence(node_id)  # Your function
    
    # Get external score
    pattern = indicator.get('pattern', '')
    hash_value = extract_hash(pattern)  # Extract hash from STIX pattern
    if hash_value:
        vt_score = fetch_vt_score(hash_value, vt_api_key)
        if vt_score is not None:
            model_confidences.append(model_conf)
            external_scores.append(vt_score)

# Compute agreement metrics
if len(model_confidences) > 0:
    pearson_r, pearson_p = pearsonr(model_confidences, external_scores)
    spearman_r, spearman_p = spearmanr(model_confidences, external_scores)
    mae = np.mean(np.abs(np.array(model_confidences) - np.array(external_scores)))
    disagreement_rate = np.mean(np.abs(np.array(model_confidences) - np.array(external_scores)) > 0.2)
    
    print(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Disagreement rate (>0.2): {disagreement_rate:.2%}")
```

**Interpretation**: 
- High correlation (>0.6) suggests good agreement
- Low MAE (<0.2) indicates similar confidence levels
- High disagreement may indicate:
  - Model needs calibration
  - External source has different threat model
  - Missing relationships in graph

### 5) Ablation Studies

**Purpose**: Quantify the contribution of each model component to understand what drives predictions.

**Steps**:
1. Define baseline configuration (full model with all components enabled).
2. Create ablated configurations, disabling one component at a time:
   - **No time decay**: Set all `time_decay_half_life` to very large values (effectively disabled)
   - **No relationship type weighting**: Set all `rel_type_weight` to `default_rel_weight`
   - **No EMA smoothing**: Set `ema_alpha` to 0
   - **No parent cap**: Set `MAX_PARENTS_PER_NODE` to very large value
   - **Single relationship type**: Remove all but one relationship type
   - **No damping**: Set `lbp_damping` to 1.0 (no updates in fixed-point)
3. For each ablated configuration:
   - Build graph from same data
   - Run inference
   - Store beliefs for all nodes
4. Compare outputs to baseline:
   - Compute per-node absolute deltas: `|belief_ablated - belief_baseline|`
   - Compute aggregate metrics: mean/median/max delta, nodes affected
   - If labels available: compute accuracy/Brier score change
5. Document impact: rank components by their contribution to output changes.

**Implementation**:
```python
import yaml
from service.sync_manager import SyncManager
from service.eventbus import EventBus

# Load baseline config
with open('config/bayes.yaml') as f:
    baseline_cfg = yaml.safe_load(f)

# Create ablated configs
ablations = {
    'no_time_decay': {
        **baseline_cfg,
        'time_decay_half_life': {k: 100000 for k in baseline_cfg['time_decay_half_life']}
    },
    'no_rel_type_weighting': {
        **baseline_cfg,
        'rel_type_weight': {},
        'default_rel_weight': baseline_cfg.get('default_rel_weight', 0.5)
    },
    'no_ema': {
        **baseline_cfg,
        'ema_alpha': 0.0
    },
    'uniform_weights': {
        **baseline_cfg,
        'rel_type_weight': {k: 0.5 for k in baseline_cfg.get('rel_type_weight', {})}
    }
}

baseline_manager = SyncManager(max_parents=5, bus=EventBus(), cfg=baseline_cfg)
baseline_manager.build_from_opencti(objects, relationships)
baseline_beliefs = baseline_manager.bayes.infer_all()

ablation_results = {}
for name, cfg in ablations.items():
    manager = SyncManager(max_parents=5, bus=EventBus(), cfg=cfg)
    manager.build_from_opencti(objects, relationships)
    beliefs = manager.bayes.infer_all()
    
    deltas = [abs(beliefs[nid] - baseline_beliefs[nid]) 
              for nid in baseline_beliefs if nid in beliefs]
    
    ablation_results[name] = {
        'mean_delta': np.mean(deltas) if deltas else 0,
        'median_delta': np.median(deltas) if deltas else 0,
        'max_delta': max(deltas) if deltas else 0,
        'nodes_affected': sum(1 for d in deltas if d > 0.01),
        'total_nodes': len(deltas)
    }

# Rank by impact
sorted_ablations = sorted(ablation_results.items(), 
                         key=lambda x: x[1]['mean_delta'], 
                         reverse=True)

print("Ablation Study Results (ranked by mean delta):")
for name, result in sorted_ablations:
    print(f"{name}:")
    print(f"  Mean delta: {result['mean_delta']:.6f}")
    print(f"  Max delta: {result['max_delta']:.6f}")
    print(f"  Nodes affected (>0.01): {result['nodes_affected']}/{result['total_nodes']}")
```

**Interpretation**: 
- Components with large deltas (>0.1) are critical for predictions
- Components with small deltas (<0.01) may be redundant or have minimal impact
- Use results to simplify model or focus tuning efforts on high-impact components

### 6) Holdout Evaluation (Ground-Truth Labels)

**Purpose**: Measure calibration and discrimination performance with labeled data using proper train/test split.

**Steps**:
1. **Prepare labeled dataset**:
   - Collect entities with ground-truth binary labels (e.g., "confirmed malicious" = 1, "benign" = 0)
   - Labels should be independent of model predictions
   - Ensure sufficient positive and negative examples
2. **Split data**:
   - **Time-based split**: Use older data for training, newer for testing (recommended for threat intel)
   - **Random split**: 70% train, 30% test (if temporal order not important)
   - Ensure both splits have similar class distributions
3. **Train/calibrate on training set**:
   - Tune priors and relationship weights on training data
   - Optionally apply calibration (Platt scaling, isotonic regression) using `validation.calibration.calibrate_confidence`
   - Store calibration parameters
4. **Evaluate on holdout set**:
   - Run inference on test set using trained parameters
   - Apply calibration if trained
   - Compute metrics:
     - **Brier Score**: Mean squared error (lower is better, 0 = perfect)
     - **ECE/MCE**: Calibration error metrics
     - **AUC-ROC**: Area under ROC curve (discrimination, higher is better, 1 = perfect)
     - **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)
     - **Accuracy**: At optimal threshold
     - **Precision/Recall/F1**: At optimal threshold
5. **Re-calibrate if needed**:
   - If ECE > 0.1, apply calibration and re-evaluate
   - Compare metrics before/after calibration

**Implementation**:
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from validation.calibration import (
    brier_score, expected_calibration_error, 
    summarize_calibration, calibrate_confidence
)
import numpy as np

# Prepare data: entities with ground-truth labels
entities = [...]  # List of entity objects
labels = [...]    # List of binary labels [0, 1] for each entity

# Split data (time-based or random)
train_entities, test_entities, train_labels, test_labels = train_test_split(
    entities, labels, test_size=0.3, random_state=42, stratify=labels
)

# Train: Build model on training data and tune parameters
train_manager = SyncManager(max_parents=5, bus=EventBus(), cfg=train_cfg)
train_manager.build_from_opencti(train_entities, train_relationships)
train_beliefs = train_manager.bayes.infer_all()

# Get training predictions
train_predictions = [train_beliefs.get(e['id'], 0.5) for e in train_entities]

# Calibrate on training set
calibrated_train_preds, cal_metrics = calibrate_confidence(
    train_predictions, train_labels, method='platt'
)

# Test: Run inference on holdout set
test_manager = SyncManager(max_parents=5, bus=EventBus(), cfg=train_cfg)  # Use same config
test_manager.build_from_opencti(test_entities, test_relationships)
test_beliefs = test_manager.bayes.infer_all()

# Get test predictions
test_predictions = [test_beliefs.get(e['id'], 0.5) for e in test_entities]

# Apply calibration from training
# (In practice, you'd save the calibration model and apply it here)
test_calibrated, _ = calibrate_confidence(test_predictions, test_labels, method='platt')

# Compute metrics
brier_uncal = brier_score(test_predictions, test_labels)
brier_cal = brier_score(test_calibrated, test_labels)
ece_uncal, mce_uncal = expected_calibration_error(test_predictions, test_labels)
ece_cal, mce_cal = expected_calibration_error(test_calibrated, test_labels)
auc_roc = roc_auc_score(test_labels, test_predictions)
pr_auc = average_precision_score(test_labels, test_predictions)

print("Holdout Evaluation Results:")
print(f"Brier Score (uncalibrated): {brier_uncal:.4f}")
print(f"Brier Score (calibrated): {brier_cal:.4f}")
print(f"ECE (uncalibrated): {ece_uncal:.4f}, MCE: {mce_uncal:.4f}")
print(f"ECE (calibrated): {ece_cal:.4f}, MCE: {mce_cal:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Compute precision/recall at optimal threshold
predicted_binary = (np.array(test_predictions) >= optimal_threshold).astype(int)
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(test_labels, predicted_binary)
recall = recall_score(test_labels, predicted_binary)
f1 = f1_score(test_labels, predicted_binary)

print(f"\nAt optimal threshold ({optimal_threshold:.4f}):")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
```

**Interpretation**:
- **Brier Score < 0.25**: Good calibration
- **ECE < 0.1**: Well-calibrated
- **AUC-ROC > 0.8**: Good discrimination
- **PR-AUC**: Better metric for imbalanced data (focus on positive class)
- **Calibration improvement**: If ECE decreases significantly after calibration, model benefits from calibration

## Project Structure

```
bayesian_opencti/
├── api/                    # Flask API server
│   └── server.py          # API endpoints and SSE streaming
├── bayes/                  # Bayesian inference engine
│   └── model.py           # Noisy-OR model and SCC solver
├── config/                 # Configuration files
│   ├── bayes.yaml         # Bayesian model parameters
│   └── settings.py        # Pydantic settings
├── dashboard/              # Web UI
│   ├── index.html         # Dashboard HTML
│   ├── app.js             # Dashboard JavaScript
│   └── styles.css         # Dashboard styles
├── docker/                 # Docker Compose for OpenCTI platform
│   ├── docker-compose.yml
│   └── .env.sample
├── scripts/                # Utility scripts
│   ├── validate_sample_data.py
│   └── mock_data_loader.py
├── service/                # Core service components
│   ├── app.py             # Main application bootstrap
│   ├── opencti_client.py  # OpenCTI API client
│   ├── sync_manager.py    # Graph sync and inference orchestration
│   ├── eventbus.py        # Event bus for SSE
│   └── logging_setup.py   # Logging configuration
├── tests/                  # Test suite
│   ├── test_api.py
│   ├── test_bayes.py
│   ├── test_integration.py
│   ├── test_validation.py
│   └── e2e/
├── validation/             # Validation and calibration
│   ├── calibration.py
│   └── data_validators.py
├── .env.example            # Environment variables template
├── .flake8                 # Flake8 configuration
├── .github/workflows/      # CI/CD workflows
├── pyproject.toml         # Project metadata and tool configs
├── pytest.ini             # Pytest configuration
├── requirements.txt        # Python dependencies
├── run.py                  # Application entry point
└── sample_data.json        # Sample STIX data for testing
```

## Assumptions and limitations

- **Proxy labels**: Proxy labels from sightings are **not** ground truth; calibration metrics are indicative only.
- **Data quality**: Confidence propagation depends on relationship coverage and quality in OpenCTI.
- **OpenCTI dependency**: OpenCTI credentials and network access are required for live data ingestion. Falls back to `sample_data.json` if unavailable.
- **Security**: API endpoints are unauthenticated; deploy behind trusted network controls in production.
- **Graph complexity**: Large graphs may require tuning of `MAX_PARENTS_PER_NODE` and solver parameters for performance.
- **Time decay**: Time decay assumes entity types have different half-lives; adjust in `config/bayes.yaml` based on your threat intelligence lifecycle.

## Troubleshooting

### OpenCTI Connection Issues

If the service cannot connect to OpenCTI:
- Check `OPENCTI_URL` and `OPENCTI_TOKEN` in `.env`
- Verify OpenCTI is running and accessible
- The service will automatically fall back to `sample_data.json` if OpenCTI is unavailable

### Graph Not Updating

- Check polling interval: `POLL_INTERVAL_SECONDS` in `.env`
- Verify OpenCTI connection is working
- Check logs for errors: `LOG_LEVEL=DEBUG` for verbose output

### Inference Not Converging

- Increase `lbp_max_iters` in `config/bayes.yaml`
- Adjust `lbp_damping` (lower = more aggressive updates)
- Check for cycles in the graph that may cause instability

### Dashboard Not Loading

- Ensure Flask server is running on port 5000
- Check browser console for JavaScript errors
- Verify SSE endpoint `/api/v1/stream` is accessible

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `flake8 . && black --check . && mypy .`
6. Submit a pull request

## License

MIT License
