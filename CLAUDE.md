# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**TurboDesigner 2.0** - A machine learning-powered design optimization tool for turbine/turbocharger systems. Uses reinforcement learning with physics-based constraints (surge/choke margins) to optimize efficiency and pressure ratio.

## Environment Setup

```bash
# Using Conda (recommended)
conda env create -f openenv.yaml
conda activate openenv

# Using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Python Version**: 3.10.5

## Directory Structure

```
turbodesigner2.0/
├── api/                 # Flask web API
│   ├── app.py          # Main application entry point
│   └── routes.py       # API route handlers
├── env/                 # RL environment module
│   ├── physics.py      # Physics calculations
│   ├── reward.py       # Reward function (efficiency, PR, constraints)
│   ├── tasks.py        # Task definitions and registry
│   ├── graders.py      # Performance grading (Pass/Fail)
│   └── config.py       # Configuration
├── tests/               # Pytest test suite
│   ├── test_api.py
│   └── test_env.py
├── inference.py         # ML inference loop
├── requirements.txt     # Python dependencies
├── openenv.yaml         # Conda environment spec
└── Dockerfile           # Container configuration
```

## Key Commands

```bash
# Run the API
python -m api.app

# Run inference
python inference.py

# Run tests
pytest tests/

# Docker
docker build -t turbodesigner2.0 .
docker run -p 5000:5000 turbodesigner2.0
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Run prediction |
| GET | `/api/tasks` | List available tasks |

## Architecture Notes

- **Reward Function**: Combines progress reward (efficiency delta), pressure ratio gating, and hard physics constraints (surge/choke penalties)
- **Physics Module**: Basic calculations (F=ma, velocity, position) - extensible for domain-specific physics
- **Task System**: RL-style task interface with `reset()`, `step()`, `is_complete()` methods
- **Grading**: Pluggable grader system with Pass/Fail baseline implementation

## Current Status

Early development - core scaffolding in place. Stubs remain for:
- ML model loading/inference
- Concrete task implementations
- Prediction logic in API
