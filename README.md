# TurboDesigner 2.0

A machine learning-powered design optimization tool.

## Project Structure

```
turbodesigner2.0/
├── api/                # Web API (Flask)
│   ├── app.py
│   └── routes.py
├── env/                # Environment module
│   ├── core_env.py
│   ├── physics.py
│   ├── reward.py
│   ├── tasks.py
│   └── graders.py
├── tests/              # Test suite
│   ├── test_api.py
│   └── test_env.py
├── inference.py        # ML inference loop
├── openenv.yaml        # Conda environment
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

### Using Conda

```bash
conda env create -f openenv.yaml
conda activate openenv
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the API

```bash
python -m api.app
```

### Run Inference

```bash
python inference.py
```

### Run Tests

```bash
pytest tests/
```

## Docker

Build and run with Docker:

```bash
docker build -t turbodesigner2.0 .
docker run -p 5000:5000 turbodesigner2.0
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Run prediction
- `GET /api/tasks` - List available tasks

## License

MIT
