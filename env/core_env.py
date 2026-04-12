from copy import deepcopy

from env import INIT_PARAMS, BOUNDS
from env.models import Action, Observation, StepInfo, safe_default_action
from env.physics import compute_physics
from env.constraints import check_constraints
from env.tasks import TASKS, FeasibilityTask, get_task


def clamp(params):
    for key in ["r2", "blade_angle", "b2", "Z"]:
        low, high = BOUNDS[key]
        params[key] = max(low, min(high, params[key]))
    params["Z"] = int(round(params["Z"]))
    return params


def normalize_action(action):
    if isinstance(action, Action):
        return action
    if isinstance(action, dict):
        try:
            return Action(**action)
        except Exception:
            return safe_default_action()
    return safe_default_action()


def apply_action(state, action):
    normalized_action = normalize_action(action)
    new = deepcopy(state)

    new["r2"] += normalized_action.delta_r2
    new["blade_angle"] += normalized_action.delta_angle
    new["b2"] += normalized_action.delta_b2
    new["Z"] += normalized_action.delta_Z

    return clamp(new)


class BladeLabEnv:

    def __init__(self, task=None, task_name=None, task_kwargs=None):
        self._state = None
        self.physics = None
        self.constraints = None
        self.prev_physics = None
        self.step_count = 0
        self.history = []
        self.task = self._resolve_task(task=task, task_name=task_name, task_kwargs=task_kwargs or {})

    def _resolve_task(self, task=None, task_name=None, task_kwargs=None):
        if task is not None:
            return task
        if task_name is not None:
            if task_kwargs:
                raise ValueError("Parameterized task construction is not supported by the static TASKS registry.")
            return get_task(task_name)
        return FeasibilityTask()

    def reset(self):
        self._state = deepcopy(INIT_PARAMS)
        self.task.reset()
        physics = compute_physics(self._state)
        constraints = check_constraints(physics)
        self.physics = physics
        self.constraints = constraints
        self.prev_physics = physics

        self.step_count = 0
        self.history = []
        return self._build_obs(physics, constraints)

    def step(self, action):
        normalized_action = normalize_action(action)
        self._state = apply_action(self._state, normalized_action)

        physics = compute_physics(self._state)
        constraints = check_constraints(physics)
        reward = self.task.compute_reward(physics, constraints, prev_physics=self.prev_physics)

        next_obs = self._build_obs(physics, constraints)

        self.history.append({
            "step": self.step_count,
            "state": self._build_obs(self.prev_physics, check_constraints(self.prev_physics)).model_dump(),
            "action": normalized_action.model_dump(),
            "reward": float(reward),
            "next_state": next_obs.model_dump(),
            "info": {
                "task": self.task.name,
                "constraints": constraints,
                "success": self.task.is_success(physics, constraints),
                "step_count": self.step_count + 1,
            },
        })

        self.physics = physics
        self.constraints = constraints
        self.prev_physics = physics
        self.step_count += 1

        done = self.task.is_done(self.step_count, physics, constraints)

        info = StepInfo(
            task=self.task.name,
            constraints=constraints,
            success=self.task.is_success(physics, constraints),
            step_count=self.step_count,
        )
        return next_obs, reward, done, info

    def _build_obs(self, physics, constraints):
        return Observation(
            efficiency=physics["efficiency"],
            pressure_ratio=physics["pressure_ratio"],
            mass_flow=physics["mass_flow"],
            feasible=constraints["feasible"],
            surge_margin=constraints["surge_margin"],
            choke_margin=constraints["choke_margin"],
            r2=self._state["r2"],
            blade_angle=self._state["blade_angle"],
            b2=self._state["b2"],
            Z=self._state["Z"],
        )

    def state(self):
        return self._build_obs(self.physics, self.constraints)

    def get_history(self):
        return self.history

    def get_trajectory(self):
        return self.history

    def close(self):
        return None
