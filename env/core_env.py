from copy import deepcopy
from env import INIT_PARAMS, BOUNDS
from env.physics import compute_physics
from env.constraints import check_constraints
from env.reward import compute_reward


def clamp(params):
    for key in ["r2", "blade_angle", "b2", "Z"]:
        low, high = BOUNDS[key]
        params[key] = max(low, min(high, params[key]))
    return params


def apply_action(state, action):
    new = deepcopy(state)

    new["r2"] += action.get("delta_r2", 0.0)
    new["blade_angle"] += action.get("delta_angle", 0.0)
    new["b2"] += action.get("delta_b2", 0.0)
    new["Z"] += int(action.get("delta_Z", 0))

    return clamp(new)


class BladeLabEnv:

    def __init__(self):
        self.state = None
        self.prev_physics = None
        self.step_count = 0
        self.history = []
        self.m_max = 1e-6   # critical for phi normalization

    def reset(self):
        self.state = deepcopy(INIT_PARAMS)

        physics = compute_physics(self.state)
        self.prev_physics = physics

        self.step_count = 0
        self.history = []

        self.m_max = physics["mass_flow"]

        return self._build_obs(physics)

    def step(self, action):
        self.state = apply_action(self.state, action)

        physics = compute_physics(self.state)

        # update max flow
        self.m_max = max(self.m_max, physics["mass_flow"])

        constraints = check_constraints(physics, self.m_max)

        reward = compute_reward(physics, self.prev_physics, constraints)

        self.history.append({
            "mass_flow": physics["mass_flow"],
            "pressure_ratio": physics["pressure_ratio"]
        })

        self.prev_physics = physics
        self.step_count += 1

        done = self.step_count >= 30

        return self._build_obs(physics), reward, done, {}

    def _build_obs(self, physics):
        return {
            "efficiency": physics["efficiency"],
            "pressure_ratio": physics["pressure_ratio"],
            "mass_flow": physics["mass_flow"],
            "r2": self.state["r2"],
            "blade_angle": self.state["blade_angle"],
            "b2": self.state["b2"],
            "Z": self.state["Z"]
        }

    def get_history(self):
        return self.history

    def get_trajectory(self):
        return self.history