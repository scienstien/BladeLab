---
title: turbodesigner2
sdk: docker
app_port: 8000
---

# TurboDesigner 2.0

Physics-driven turbomachinery design environment with both a local Flask API and an OpenEnv-compatible FastAPI server.

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
├── client.py           # OpenEnv client wrapper
├── models.py           # OpenEnv-facing models
├── server/             # OpenEnv FastAPI server
├── inference.py        # Baseline inference loop
├── openenv.yaml        # OpenEnv manifest
├── pyproject.toml      # OpenEnv packaging metadata
├── uv.lock             # uv lockfile
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

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

All runtime entrypoints import the template through `python-dotenv`, so copying the template is enough for the local scripts and `flask` app to pick up those values.

Variables used by the current code paths:

- `OPENAI_API_KEY`: required only when using the OpenAI-backed policy paths
- `MODEL_NAME`: optional; defaults to `gpt-4.1-mini`
- `API_BASE_URL`: optional; takes precedence over `OPENAI_BASE_URL` when both are set
- `OPENAI_BASE_URL`: optional fallback for OpenAI-compatible endpoints
- `HF_TOKEN`: optional token for Hugging Face Space deployment or push workflows

You only need OpenAI credentials when:

- calling the API with `policy_type="openai"`
- running `python inference.py --openai`

Heuristic policy flows and the current test suite do not require `OPENAI_API_KEY`.

## Usage

### Run the OpenEnv Server

```bash
python -m server.app --port 8000
```

### Run the Flask API

```bash
python -m api.app
```

### Run Baseline Inference

```bash
python inference.py
```

### Run Tests

```bash
python -m pytest tests/
```

## Docker

Build and run with Docker:

```bash
docker build -t turbodesigner2.0 .
docker run -p 8000:8000 turbodesigner2.0
```

To use the OpenAI-backed policy in Docker, pass env vars at runtime instead of
baking secrets into the image:

```bash
docker run --env-file .env -p 8000:8000 turbodesigner2.0
```

For OpenEnv validation, the image should expose the FastAPI app on port `8000`.

## Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Run prediction
- `GET /api/tasks` - List available tasks
- `POST /reset` - OpenEnv reset endpoint
- `POST /step` - OpenEnv step endpoint
- `GET /state` - OpenEnv state endpoint
- `GET /schema` - OpenEnv action/observation schema endpoint

## License

MIT
