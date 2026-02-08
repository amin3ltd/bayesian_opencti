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

## Dashboard Features

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

## Configuration Reference

### `config/bayes.yaml`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lbp_damping` | Damping factor for cyclic SCC | 0.5 |
| `lbp_epsilon` | Convergence threshold | 1e-6 |
| `lbp_max_iters` | Max fixed-point iterations | 100 |
| `rel_type_weight` | Weights by relationship type | - |
| `default_rel_weight` | Default edge weight | 0.5 |
| `ema_alpha` | Smoothing factor | 0.3 |
| `confidence_push_delta_min` | Min delta to push | 0.01 |
| `time_decay_half_life` | Time decay by entity type | - |

## License

MIT License
