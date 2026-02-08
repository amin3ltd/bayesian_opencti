# Bayesian Confidence Scoring for OpenCTI

## Abstract

This academic research project develops a Bayesian confidence propagation system for OpenCTI (Open Cyber Threat Intelligence). The methodology combines Noisy-OR inference with damped fixed-point updates for cyclic graphs to estimate posterior confidence for threat intelligence entities while preserving interpretability through explicit edge weights and parent contributions.

## Research Objectives

- Formalize confidence propagation for heterogeneous threat intelligence graphs.
- Provide real-time posterior updates with transparent attribution paths.
- Evaluate output quality with statistical consistency checks and calibration metrics.

## System Features

- **Hybrid Inference**: Exact Noisy-OR updates for acyclic components, damped fixed-point for cyclic strongly connected components.
- **Real-time Updates**: Live confidence propagation via Server-Sent Events (SSE).
- **Web Dashboard**: Interactive graph visualization with Cytoscape.js, showing nodes colored by confidence levels.
- **OpenCTI Integration**: Polls OpenCTI for changes and pushes updated confidence scores back.
- **Configurable**: Extensive configuration via `config/bayes.yaml` for relation weights, time decay, smoothing, etc.

## Prerequisites

- Python 3.8+
- An OpenCTI instance (local or remote)
- Git

## Installation

### Option 1: Direct Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amin3ltd/bayesian_opencti.git
   cd bayesian_opencti
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenCTI credentials
   ```

5. **Configure Bayesian parameters**
   Edit `config/bayes.yaml` as needed

6. **Run the application**
   ```bash
   python run.py
   ```

### Option 2: Docker Deployment

```bash
cd docker
cp .env.sample .env
docker-compose up -d
```

## Access

- **Dashboard**: http://localhost:5000
- **API**: http://localhost:5000/api

## Result Validation

This section documents methods and frameworks for validating the confidence scores produced by the Bayesian inference engine.

### 1. Statistical Consistency Checks

Basic mathematical validation that outputs are internally consistent:

| Check | Description | Valid Range |
|-------|-------------|-------------|
| **Bounds Check** | All probabilities are valid | [0, 1] |
| **Normalization** | Beliefs sum appropriately | N/A for Noisy-OR |
| **Convergence** | Fixed-point iterations stabilize | Δ < ε |
| **Monotonicity** | Adding evidence should not decrease confidence | Conditional |

**Implementation:**
```python
# bayes/model.py already includes:
EPS = 1e-9
def clamp01(x: float) -> float:
    return max(EPS, min(1.0 - EPS, float(x)))
```

### 2. Calibration Validation

Calibration measures how well confidence scores match actual accuracy.

**Recommended Framework**: [Reliability Diagrams](https://scikit-learn.org/stable/modules/calibration.html)

**Methodology:**
1. Bin predictions by confidence (0-10%, 10-20%, ..., 90-100%)
2. Track actual outcomes (true/false positives from threat intel)
3. Plot predicted vs actual accuracy

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Calculate calibration curve
prob_true, prob_pred = calibration_curve(
    actual_labels, 
    predicted_confidences, 
    n_bins=10
)

# Perfectly calibrated = diagonal line
```

**Frameworks:**
- [scikit-learn](https://scikit-learn.org/stable/) - Calibration curves, Brier score
- [SciPy](https://scipy.org/) - Statistical tests for calibration
- [MLmetrics](https://github.com/mlmetrics/mlmetrics) - Calibration error metrics

**Utilities (in-repo):**
- `validate_calibration_inputs` ensures bounds and binary outcomes.
- `summarize_calibration` provides aggregate calibration statistics for reporting.

#### Sample Data Calibration Summary (proxy labels)

Using `sample_data.json`, we ran `validation.calibration.summarize_calibration`
on objects with confidence scores (excluding relationship objects). Because the
sample data does not include ground-truth outcomes, we used sightings as a proxy
label: outcome = 1 for objects referenced by `sighting_of_ref` or
`where_sighted_refs`, else 0. This proxy is sparse and should be interpreted as
an exploratory sanity check rather than a definitive calibration study.

- Eligible objects (non-relationship with confidence): 11
- Positive rate (proxy): 0.0909
- Mean confidence: 0.7227
- Brier score: 0.6202
- ECE: 0.7773
- MCE: 0.8667
- n_bins: 5

### 3. Cross-Validation with External Feeds

Validate against independent threat intelligence sources:

| Source | Use Case |
|--------|----------|
| [VirusTotal](https://developers.virustotal.com/reference) | File/URL/IP reputation |
| [AlienVault OTX](https://otx.alienvault.com/api) | Threat pulses |
| [MISP](https://www.misp-project.org/api/) | Event correlation |
| [MITRE ATT&CK](https://attack.mitre.org/docs/attack-api/ATTACK_API_v10.0.pdf) | Technique validation |

**Validation Rules:**
1. High confidence (>80%) on known malware → True Positive
2. High confidence (>80%) on benign → False Positive
3. Low confidence (<20%) on confirmed threats → False Negative

### 4. Temporal Validation

Track confidence evolution over time:

```python
# From service/sync_manager.py - history tracking
history = self._history.get(nid, [])
# Validates: confidence should NOT oscillate wildly
# Monotonic decay expected for stale indicators
```

**Anomaly Detection:**
- Sudden confidence spikes → potential false positive
- Unexpected drops → may indicate data staleness

### 5. Graph Consistency Validation

Validate the Bayesian network structure:

| Check | Description |
|-------|-------------|
| **Acyclicity** | No cycles (except intentional SCCs) |
| **Connectivity** | All nodes reachable from evidence |
| **Parent Cap** | No node exceeds max_parents |
| **Edge Weights** | All weights in [0, 1] |

### Recommended Validation Workflow

```
Confidence Output
        ↓
[Statistical Checks] → FAIL → Review model parameters
        ↓ PASS
[Calibration Analysis] → FAIL → Recalibrate priors/weights
        ↓ PASS
[Cross-Reference Feed] → FAIL → Flag for review
        ↓ PASS
[Time-Series Check] → FAIL → Investigate anomalies
        ↓
[Store validated results]
```

### Metrics & Frameworks

| Metric | Framework | Purpose |
|--------|-----------|---------|
| **Brier Score** | scikit-learn | Calibration quality (lower = better) |
| **Expected Calibration Error (ECE)** | custom | Weighted calibration error |
| **Log Loss** | scikit-learn / MLmetrics | Cross-entropy loss |
| **AUC-ROC** | scikit-learn | Discrimination ability |
| **Precision-Recall** | scikit-learn | Imbalanced threat data |

### Brier Score Implementation

```python
import numpy as np

def brier_score(predictions, outcomes):
    """Lower is better (0 = perfect, 1 = worst)"""
    return np.mean((predictions - outcomes) ** 2)

# Typical values for threat intel:
# Well-calibrated: 0.10 - 0.20
# Poorly calibrated: 0.30+
```

### Validation Checklist

- [ ] All belief values within [0, 1]
- [ ] Convergence achieved (Δ < ε)
- [ ] Brier score tracked over time
- [ ] Cross-references with external feeds
- [ ] Anomalies flagged and reviewed
- [ ] Model recalibrated quarterly

### Validation Outcomes (Sample Data)

The following results were computed on `sample_data.json` using STIX 2.1 parsing
and the calibration utilities in `validation/calibration.py`.

- STIX objects: 47
- STIX parse errors: 0
- Confidence present: 37
- Confidence missing: 10
- Confidence non-numeric: 0
- Confidence out of range: 0
- Mean confidence (0-100): 77.7

### Correctness Audit (Resolved)

Logical issues identified during review and addressed in the current `main` branch:

- **STIX relationship semantics**: relationship objects now use `relationship_type`
  when present, avoiding incorrect default weighting when `type == "relationship"`.
- **Missing confidence handling**: OpenCTI objects preserve `None` confidence so
  default priors are applied instead of forcing zero confidence.
- **Time decay application**: node priors now apply time decay using
  `updated_at`, `modified`, or `created` timestamps with case-insensitive type
  matching for configuration keys.
- **Case-insensitive weights**: relationship type weights and time-decay maps are
  normalized to ensure consistent behavior across data sources.

## Architecture

- `bayes/model.py`: Custom Bayesian network with hybrid inference
- `service/sync_manager.py`: Manages graph building, inference, and OpenCTI sync
- `api/server.py`: Flask API with SSE for live updates
- `dashboard/`: Static web UI

## Testing

```bash
python3 -m pytest
```

## Configuration Reference

See `config/bayes.yaml` for inference parameters, relationship weights, and time decay settings.

## License

MIT License
