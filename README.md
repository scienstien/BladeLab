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

## Runtime Configuration

OpenAI-backed inference is configured from environment variables at runtime.
The project keeps secrets in a local `.env` file or in the host environment.
`.env` is ignored by git and should not be committed.

Start from the checked-in `.env.example` template and create a local `.env`:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Variables used by the current code paths:

- `OPENAI_API_KEY`: required only when using the OpenAI-backed policy paths
- `MODEL_NAME`: optional; defaults to `gpt-4.1-mini`
- `API_BASE_URL`: optional; takes precedence over `OPENAI_BASE_URL` when both are set
- `OPENAI_BASE_URL`: optional fallback for OpenAI-compatible endpoints

You only need OpenAI credentials when:

- calling the API with `policy_type="openai"`
- running `python inference.py --openai`

Heuristic policy flows and the current test suite do not require `OPENAI_API_KEY`.

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

To use the OpenAI-backed policy in Docker, pass env vars at runtime instead of
baking secrets into the image:

```bash
docker run --env-file .env -p 5000:5000 turbodesigner2.0
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Run prediction
- `GET /api/tasks` - List available tasks

## License

MIT
