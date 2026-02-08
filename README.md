# Bayesian Confidence Scoring for OpenCTI

This project implements a Bayesian network-based confidence scoring system for OpenCTI (Open Cyber Threat Intelligence). It uses a custom hybrid inference engine that combines Noisy-OR logic with damped fixed-point iteration for cyclic graphs to compute posterior probabilities for threat intelligence entities.

## Features

- **Hybrid Inference**: Exact Noisy-OR updates for acyclic components, damped fixed-point for cyclic strongly connected components.
- **Real-time Updates**: Live confidence propagation via Server-Sent Events (SSE).
- **Web Dashboard**: Interactive graph visualization with Cytoscape.js, showing nodes colored by confidence levels.
- **OpenCTI Integration**: Polls OpenCTI for changes and pushes updated confidence scores back.
- **Configurable**: Extensive configuration via `config/bayes.yaml` for relation weights, time decay, smoothing, etc.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` (see `.env.example`):
   - `OPENCTI_URL`: Your OpenCTI instance URL
   - `OPENCTI_TOKEN`: API token
   - `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
   - `POLL_INTERVAL_SECONDS`: Polling interval
   - `MAX_PARENTS_PER_NODE`: Max parents per node in the Bayesian network
4. Configure Bayesian parameters in `config/bayes.yaml`.
5. Run: `python run.py`

## Configuration

### `config/bayes.yaml`

- `lbp_damping`: Damping factor for cyclic SCC fixed-point (0-1)
- `lbp_epsilon`: Convergence threshold
- `lbp_max_iters`: Max iterations for fixed-point
- `rel_type_weight`: Weights for different relationship types
- `default_rel_weight`: Fallback weight
- `ema_alpha`: Exponential moving average alpha for smoothing
- `confidence_push_delta_min`: Min change to push updates
- `time_decay_half_life`: Half-lives for time decay by entity type
- `rel_conf_fallback`: Fallback confidence for relationships
- `report_object_min`: Min confidence for report-object links

### Dashboard

- Accessible at `http://localhost:5000`
- Graph view with confidence-based coloring
- Node details on click: priors, posteriors, contributions, paths
- History sparkline for confidence evolution
- Recompute button to force full inference

## Architecture

- `bayes/model.py`: Custom Bayesian network with hybrid inference
- `service/sync_manager.py`: Manages graph building, inference, and OpenCTI sync
- `api/server.py`: Flask API with SSE for live updates
- `dashboard/`: Static web UI

## Testing

Run tests: `pytest tests/`

## Requirements

- Python 3.8+
- OpenCTI instance
- Dependencies in `requirements.txt`
