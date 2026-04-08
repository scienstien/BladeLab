"""OpenEnv client for the TurboDesigner environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import TurboDesignerAction, TurboDesignerObservation, TurboDesignerState


class TurboDesignerEnv(
    EnvClient[TurboDesignerAction, TurboDesignerObservation, TurboDesignerState]
):
    """Thin client wrapper around the OpenEnv HTTP/WebSocket server."""

    def _step_payload(self, action: TurboDesignerAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[TurboDesignerObservation]:
        observation = TurboDesignerObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> TurboDesignerState:
        return TurboDesignerState(**payload)
