# Bayesian Confidence Scoring for OpenCTI

This project implements a Bayesian network-based confidence scoring system for OpenCTI (Open Cyber Threat Intelligence). It uses a custom hybrid inference engine that combines Noisy-OR logic with damped fixed-point iteration for cyclic graphs to compute posterior probabilities for threat intelligence entities.

## Features

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
   
   Required environment variables:
   - `OPENCTI_URL`: Your OpenCTI instance URL
   - `OPENCTI_TOKEN`: API token for authentication
   - `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
   - `POLL_INTERVAL_SECONDS`: Polling interval in seconds
   - `MAX_PARENTS_PER_NODE`: Max parents per node in the Bayesian network

5. **Configure Bayesian parameters**
   Edit `config/bayes.yaml` to customize:
   - `lbp_damping`: Damping factor for cyclic SCC fixed-point (0-1)
   - `lbp_epsilon`: Convergence threshold
   - `lbp_max_iters`: Max iterations for fixed-point
   - `rel_type_weight`: Weights for different relationship types
   - `default_rel_weight`: Fallback weight
   - `ema_alpha`: Exponential moving average alpha for smoothing
   - `confidence_push_delta_min`: Min change to push updates
   - `time_decay_half_life`: Half-lives for time decay by entity type

6. **Run the application**
   ```bash
   python run.py
   ```

### Option 2: Docker Deployment

1. **Clone and navigate to the project**
   ```bash
   git clone https://github.com/amin3ltd/bayesian_opencti.git
   cd bayesian_opencti/docker
   ```

2. **Configure environment**
   ```bash
   cp .env.sample .env
   # Edit .env with your configuration
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

## Access

- **Dashboard**: http://localhost:5000
- **API**: http://localhost:5000/api

## Data Validation

This section documents how data is validated and processed from OpenCTI.

### Supported Entity Types

| Entity Type | Default Prior (%) | Description |
|-------------|-------------------|-------------|
| `Indicator` | 70 | Observable indicators of compromise |
| `Malware` | 50 | Malware families and instances |
| `Threat-Actor` | 60 | Threat actor groups and individuals |
| `Campaign` | 55 | Named campaigns |
| `Intrusion-Set` | 55 | Intrusion sets |
| `Attack-Pattern` | 65 | MITRE ATT&CK patterns |
| `Report` | 80 | Threat reports (authoritative) |
| `Identity` | 75 | Organizations and individuals |
| `Infrastructure` | 45 | C2 and other infrastructure |
| `Course-of-Action` | 85 | Mitigation and response actions |

### Confidence Values

- **Input Range**: 0-100 (integer percentage)
- **Internal Range**: 0.0-1.0 (floating point)
- **Clamping**: Values are clamped to [ε, 1-ε] where ε = 1e-9 to avoid numerical instability

### Supported Relationship Types

| Relationship Type | Default Weight | Description |
|------------------|----------------|-------------|
| `indicates` | 0.85 | Indicator points to entity |
| `attributed-to` | 0.65 | Attribution to actor/threat group |
| `uses` | 0.35 | Entity uses another (e.g., malware uses infrastructure) |
| `targets` | 0.45 | Entity targets a victim |
| `object` | 0.50 | Report contains object reference |
| `delivers` | 0.50 | Malware delivery relationship |

### Relationship Validation Rules

1. **Self-loops are rejected**: Relationships where `source_ref == target_ref` are ignored
2. **Parent cap enforcement**: Each node has a maximum of `MAX_PARENTS_PER_NODE` incoming edges (default: 5). The strongest parents are retained based on edge weights
3. **Missing nodes**: Edges referencing non-existent nodes are skipped
4. **Weight calculation**: Edge weights = `relationship_confidence × type_weight`

### Time Decay Configuration

Confidence values decay over time based on entity type. Configure in `config/bayes.yaml`:

```yaml
time_decay_half_life:
  Indicator: 60      # Days
  Report: 90
  Malware: 180
  Intrusion-Set: 240
  Threat-Actor-Individual: 240
```

The decay formula: `confidence = original × 0.5^(age_days / half_life)`

### Data Input Format (STIX 2.1)

The system accepts STIX 2.1 bundle format:

```json
{
  "type": "bundle",
  "id": "bundle--uuid",
  "objects": [
    {
      "type": "indicator",
      "spec_version": "2.1",
      "id": "indicator--uuid",
      "created": "2024-01-15T12:00:00Z",
      "modified": "2024-01-15T12:00:00Z",
      "name": "Malicious Domain",
      "confidence": 80
    },
    {
      "type": "relationship",
      "spec_version": "2.1",
      "id": "relationship--uuid",
      "relationship_type": "indicates",
      "source_ref": "indicator--uuid",
      "target_ref": "malware--uuid",
      "confidence": 70
    }
  ]
}
```

### Validation Pipeline

```
OpenCTI API Response
        ↓
[Fetch STIX Objects & Relationships]
        ↓
[Validate required fields (id, type, name)]
        ↓
[Apply default priors by entity type]
        ↓
[Build directed graph (skip invalid edges)]
        ↓
[Apply parent cap (top-k strongest)]
        ↓
[Run Bayesian inference (Noisy-OR + LBP)]
        ↓
[Apply time decay (if timestamp available)]
        ↓
[Push confidence updates to OpenCTI]
```

### Dashboard Features

- Interactive graph visualization with confidence-based coloring
- Node details on click: priors, posteriors, contributions, paths
- History sparkline for confidence evolution
- Recompute button to force full inference

## Architecture

- `bayes/model.py`: Custom Bayesian network with hybrid inference
- `service/sync_manager.py`: Manages graph building, inference, and OpenCTI sync
- `api/server.py`: Flask API with SSE for live updates
- `dashboard/`: Static web UI

## Testing

```bash
pytest tests/
```

### Test Coverage

- **Small network**: Basic Noisy-OR inference with single parent
- **Cyclic network**: Damped fixed-point convergence for cycles
- **Chain network**: Multi-hop propagation through the graph

## Configuration Reference

### `config/bayes.yaml`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lbp_damping` | Damping factor for cyclic SCC | 0.55 |
| `lbp_epsilon` | Convergence threshold | 1e-4 |
| `lbp_max_iters` | Max fixed-point iterations | 100 |
| `rel_type_weight` | Weights by relationship type | - |
| `default_rel_weight` | Default edge weight | 0.5 |
| `ema_alpha` | Smoothing factor | 0.35 |
| `confidence_push_delta_min` | Min delta to push | 2 |
| `time_decay_half_life` | Time decay by entity type | - |

## License

MIT License
