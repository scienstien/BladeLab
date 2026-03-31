from copy import deepcopy
from env.config import INIT_PARAMS, BOUNDS
from env.physics import compute_physics
from env.constraints import check_constraints
from env.reward import compute_reward

def clamp(params):
    for key in ["radius", "blade_angle", "thickness"]:
        low, high = BOUNDS[key]
        params[key] = max(low, min(high, params[key]))
    return params

def apply_action(state, action):
    new = deepcopy(state)

    new["radius"] += action.get("delta_radius", 0.0)
    new["blade_angle"] += action.get("delta_angle", 0.0)
    new["thickness"] += action.get("delta_thickness", 0.0)

    return clamp(new)

class BladeLabEnv:

    def __init__(self):
        self.state = None
        self.prev_physics = None
        self.step_count = 0
        self.history = []

    def reset(self):
        self.state = deepcopy(INIT_PARAMS)
        self.prev_physics = compute_physics(self.state)
        self.step_count = 0
        self.history = []

        return self._build_obs(self.prev_physics)

    def step(self, action):
        self.state = apply_action(self.state, action)

        physics = compute_physics(self.state)
        constraints = check_constraints(physics)

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
            "radius": self.state["radius"],
            "blade_angle": self.state["blade_angle"],
            "thickness": self.state["thickness"]
        }
    def get_history(self):
        return self.history

    def get_trajectory(self):
        """Return the trajectory of mass_flow and pressure_ratio."""
        return self.history