from copy import deepcopy
from env import INIT_PARAMS, BOUNDS
from env.physics import compute_physics
from env.constraints import check_constraints
from env.tasks import DEFAULT_TASKS, FeasibilityTask


def clamp(params):
    for key in ["r2", "blade_angle", "b2", "Z"]:
        low, high = BOUNDS[key]
        params[key] = max(low, min(high, params[key]))
    params["Z"] = int(round(params["Z"]))
    return params


def apply_action(state, action):
    new = deepcopy(state)

    new["r2"] += action.get("delta_r2", 0.0)
    new["blade_angle"] += action.get("delta_angle", 0.0)
    new["b2"] += action.get("delta_b2", 0.0)
    new["Z"] += int(action.get("delta_Z", 0))

    return clamp(new)


class BladeLabEnv:

    def __init__(self, task=None, task_name=None, task_kwargs=None):
        self.state = None
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
            return DEFAULT_TASKS.create(task_name, **(task_kwargs or {}))
        return FeasibilityTask()

    def reset(self):
        self.state = deepcopy(INIT_PARAMS)
        self.task.reset()
        physics = compute_physics(self.state)
        constraints = check_constraints(physics)
        self.physics = physics
        self.constraints = constraints
        self.prev_physics = physics

        self.step_count = 0
        self.history = []
        return self._build_obs(physics, constraints)

    def step(self, action):
        self.state = apply_action(self.state, action)

        physics = compute_physics(self.state)
        constraints = check_constraints(physics)
        reward = self.task.compute_reward(physics, constraints, prev_physics=self.prev_physics)

        self.history.append({
            "mass_flow": physics["mass_flow"],
            "pressure_ratio": physics["pressure_ratio"],
            "efficiency": physics["efficiency"],
            "feasible": constraints["feasible"],
        })

        self.physics = physics
        self.constraints = constraints
        self.prev_physics = physics
        self.step_count += 1

        done = self.task.is_done(self.step_count, physics, constraints)

        info = {
            "task": self.task.name,
            "constraints": constraints,
            "success": self.task.is_success(physics, constraints),
        }
        return self._build_obs(physics, constraints), reward, done, info

    def _build_obs(self, physics, constraints):
        return {
            "efficiency": physics["efficiency"],
            "pressure_ratio": physics["pressure_ratio"],
            "mass_flow": physics["mass_flow"],
            "feasible": constraints["feasible"],
            "surge_margin": constraints["surge_margin"],
            "choke_margin": constraints["choke_margin"],
            "r2": self.state["r2"],
            "blade_angle": self.state["blade_angle"],
            "b2": self.state["b2"],
            "Z": self.state["Z"],
        }

    def get_history(self):
        return self.history

    def get_trajectory(self):
        return self.history
