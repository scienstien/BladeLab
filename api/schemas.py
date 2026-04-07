"""Pydantic request/response schemas for TurboDesigner 2.0 API"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator, field_validator

from env.models import Action, Observation
from env.tasks import TASKS

VALID_TASKS = list(TASKS.keys())

# =============================================================================
# Request Schemas
# =============================================================================

class PredictRequest(BaseModel):
    """Request schema for single-step prediction endpoint"""
    observation: Observation
    task_name: str = "feasibility"
    policy_type: Literal["heuristic", "openai"] = "heuristic"
    model_name: Optional[str] = None

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"invalid task '{v}'. Valid tasks: {VALID_TASKS}")
        return v

    @model_validator(mode="after")
    def validate_openai_model_name(self) -> "PredictRequest":
        if self.policy_type == "openai" and not self.model_name:
            raise ValueError("model_name is required when policy_type is 'openai'")
        return self


class RolloutRequest(BaseModel):
    """Request schema for rollout/trajectory endpoint"""
    task_name: str
    policy_type: Literal["heuristic", "openai"]
    max_steps: int = Field(default=100, gt=0)
    model_name: Optional[str] = None

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"invalid task '{v}'. Valid tasks: {VALID_TASKS}")
        return v

    @model_validator(mode="after")
    def validate_openai_model_name(self) -> "RolloutRequest":
        if self.policy_type == "openai" and not self.model_name:
            raise ValueError("model_name is required when policy_type is 'openai'")
        return self


class EvaluateRequest(BaseModel):
    """Request schema for multi-episode evaluation endpoint"""
    task_name: str
    policy_type: Literal["heuristic", "openai"] = "heuristic"
    num_episodes: int = Field(default=10, gt=0)
    max_steps: Optional[int] = Field(default=None, gt=0)
    model_name: Optional[str] = None

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"invalid task '{v}'. Valid tasks: {VALID_TASKS}")
        return v

    @model_validator(mode="after")
    def validate_openai_model_name(self) -> "EvaluateRequest":
        if self.policy_type == "openai" and not self.model_name:
            raise ValueError("model_name is required when policy_type is 'openai'")
        return self


# =============================================================================
# Response Schemas
# =============================================================================

class PredictResponse(BaseModel):
    """Response schema for single-step prediction endpoint"""
    action: dict
    policy_type: str
    observation: dict


class RolloutResponse(BaseModel):
    """Response schema for rollout/trajectory endpoint"""
    trajectory: List[dict]
    total_reward: float
    final_state: dict
    success: bool
    steps: int
    scores: dict


class EvaluateResponse(BaseModel):
    """Response schema for multi-episode evaluation endpoint"""
    task_name: str
    policy_type: str
    reward_mean: float
    reward_variance: float
    pr_mean: float
    pr_variance: float
    efficiency_mean: float
    efficiency_variance: float
    mass_flow_mean: float
    mass_flow_variance: float
    num_episodes: int
