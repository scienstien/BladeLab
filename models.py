"""OpenEnv-facing models for the TurboDesigner environment."""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class TurboDesignerAction(Action):
    """Action schema exposed through the OpenEnv server."""

    delta_r2: float = Field(default=0.0, ge=-0.006, le=0.006)
    delta_angle: float = Field(default=0.0, ge=-4.0, le=4.0)
    delta_b2: float = Field(default=0.0, ge=-0.002, le=0.002)
    delta_Z: int = Field(default=0, ge=-2, le=2)


class TurboDesignerObservation(Observation):
    """Observation schema exposed through the OpenEnv server."""

    efficiency: float = Field(default=0.0)
    pressure_ratio: float = Field(default=0.0)
    mass_flow: float = Field(default=0.0)
    feasible: bool = Field(default=False)
    surge_margin: float = Field(default=0.0)
    choke_margin: float = Field(default=0.0)
    r2: float = Field(default=0.0)
    blade_angle: float = Field(default=0.0)
    b2: float = Field(default=0.0)
    Z: int = Field(default=0)
    task_name: str = Field(default="feasibility")
    success: bool = Field(default=False)


class TurboDesignerState(State):
    """Environment state exposed through the OpenEnv server."""

    task_name: str = Field(default="feasibility")
    success: bool = Field(default=False)


class TurboDesignerReward(BaseModel):
    """Typed reward model for submission-time schema checks."""

    value: float = Field(..., description="Scalar reward value")
