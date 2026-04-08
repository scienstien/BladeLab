"""OpenEnv wrapper around the internal BladeLab environment."""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from env.core_env import BladeLabEnv
from env.models import StepInfo
from env.tasks import TASKS
from models import TurboDesignerAction, TurboDesignerObservation, TurboDesignerReward, TurboDesignerState


class TurboDesignerEnvironment(Environment):
    """OpenEnv-compatible wrapper for the TurboDesigner RL environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._task_name = "feasibility"
        self._env = BladeLabEnv(task_name=self._task_name)
        self._state = TurboDesignerState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=self._task_name,
            success=False,
        )

    def _ensure_task(self, task_name: str) -> str:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task_name: {task_name}")
        return task_name

    def _build_observation(
        self,
        observation,
        info: StepInfo | None,
    ) -> TurboDesignerObservation:
        payload = observation.model_dump()
        return TurboDesignerObservation(
            **payload,
            task_name=self._task_name,
            success=bool(info.success) if info is not None else False,
            metadata={
                "task": self._task_name,
                "step_count": self._state.step_count,
                "constraints": self._env.constraints or {},
            },
        )

    def reset(self, seed=None, episode_id=None, task_name="feasibility", **kwargs) -> TurboDesignerObservation:
        del seed, kwargs
        self._task_name = self._ensure_task(task_name)
        self._env = BladeLabEnv(task_name=self._task_name)
        observation = self._env.reset()
        self._state = TurboDesignerState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._task_name,
            success=False,
        )
        return self._build_observation(observation, info=None)

    def step(self, action: TurboDesignerAction, timeout_s=None, **kwargs):
        del timeout_s, kwargs
        if self._env._state is None:
            self.reset(task_name=self._task_name)
        observation, reward, done, info = self._env.step(action.model_dump())
        self._state.step_count = self._env.step_count
        self._state.success = bool(info.success)
        obs = self._build_observation(observation, info=info)
        reward_obj = TurboDesignerReward(value=float(reward))
        return obs, reward_obj, done, info

    @property
    def state(self) -> TurboDesignerState:
        return self._state
