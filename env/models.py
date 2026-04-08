from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    efficiency: float
    pressure_ratio: float
    mass_flow: float
    feasible: bool
    surge_margin: float
    choke_margin: float
    r2: float
    blade_angle: float
    b2: float
    Z: int


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta_r2: float = Field(default=0.0, ge=-0.006, le=0.006)
    delta_angle: float = Field(default=0.0, ge=-4.0, le=4.0)
    delta_b2: float = Field(default=0.0, ge=-0.002, le=0.002)
    delta_Z: int = Field(default=0, ge=-2, le=2)


class StepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    success: bool
    constraints: dict
    step_count: int


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float


def safe_default_action() -> Action:
    return Action()
