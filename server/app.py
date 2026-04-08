"""FastAPI application exposing the TurboDesigner environment via OpenEnv."""

from openenv.core.env_server.http_server import create_app

from models import TurboDesignerAction, TurboDesignerObservation
from server.turbodesigner_environment import TurboDesignerEnvironment


app = create_app(
    TurboDesignerEnvironment,
    TurboDesignerAction,
    TurboDesignerObservation,
    env_name="turbodesigner2",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.port == 7860:
        main()
    else:
        main(port=args.port)
