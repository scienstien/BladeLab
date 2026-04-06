"""Concrete optimization tasks for the environment."""

from env.config import CHOKE_LIMIT, MAX_STEPS, PR_TARGET, SURGE_LIMIT
from env.reward import efficiency_bonus, infeasibility_penalty, quadratic_target_penalty


class Task:
    def __init__(self, name, description="", max_steps=MAX_STEPS):
        self.name = name
        self.description = description
        self.max_steps = max_steps

    def reset(self):
        return None

    def compute_reward(self, physics, constraints, prev_physics=None):
        raise NotImplementedError

    def is_success(self, physics, constraints):
        raise NotImplementedError

    def is_done(self, step_count, physics, constraints):
        return step_count >= self.max_steps or self.is_success(physics, constraints)


class FeasibilityTask(Task):
    def __init__(self, surge_limit=SURGE_LIMIT, choke_limit=CHOKE_LIMIT, max_steps=MAX_STEPS):
        super().__init__(
            name="feasibility",
            description="Keep the design within surge and choke bounds.",
            max_steps=max_steps,
        )
        self.surge_limit = surge_limit
        self.choke_limit = choke_limit

    def compute_reward(self, physics, constraints, prev_physics=None):
        reward = -0.01
        if not constraints["feasible"]:
            reward -= 10.0 + infeasibility_penalty(constraints, margin_weight=20.0)
        if constraints["feasible"]:
            reward += 2.0
            reward += 0.5 * min(constraints["surge_margin"], constraints["choke_margin"])
        return reward

    def is_success(self, physics, constraints):
        return constraints["feasible"]


class TargetPRTask(Task):
    def __init__(
        self,
        target_pr=PR_TARGET,
        pr_tolerance=0.05,
        surge_limit=SURGE_LIMIT,
        choke_limit=CHOKE_LIMIT,
        max_steps=MAX_STEPS,
    ):
        super().__init__(
            name="target_pr",
            description="Stay feasible while matching the target pressure ratio.",
            max_steps=max_steps,
        )
        self.target_pr = target_pr
        self.pr_tolerance = pr_tolerance
        self.surge_limit = surge_limit
        self.choke_limit = choke_limit

    def compute_reward(self, physics, constraints, prev_physics=None):
        pr_error = abs(physics["pressure_ratio"] - self.target_pr)
        reward = -0.01

        if not constraints["feasible"]:
            reward -= 12.0 + infeasibility_penalty(constraints, margin_weight=25.0)
        else:
            reward += 3.0

        reward -= quadratic_target_penalty(physics["pressure_ratio"], self.target_pr, weight=8.0)
        if pr_error <= self.pr_tolerance and constraints["feasible"]:
            reward += 2.0

        return reward

    def is_success(self, physics, constraints):
        return constraints["feasible"] and abs(physics["pressure_ratio"] - self.target_pr) <= self.pr_tolerance


class TargetPREfficiencyTask(TargetPRTask):
    def __init__(
        self,
        target_pr=PR_TARGET,
        pr_tolerance=0.05,
        min_efficiency=0.75,
        surge_limit=SURGE_LIMIT,
        choke_limit=CHOKE_LIMIT,
        max_steps=MAX_STEPS,
    ):
        super().__init__(
            target_pr=target_pr,
            pr_tolerance=pr_tolerance,
            surge_limit=surge_limit,
            choke_limit=choke_limit,
            max_steps=max_steps,
        )
        self.name = "target_pr_efficiency"
        self.description = "Stay feasible, match target pressure ratio, and maximize efficiency."
        self.min_efficiency = min_efficiency

    def compute_reward(self, physics, constraints, prev_physics=None):
        reward = super().compute_reward(physics, constraints, prev_physics=prev_physics)

        if constraints["feasible"]:
            reward += efficiency_bonus(physics, weight=5.0)
            if physics["efficiency"] >= self.min_efficiency:
                reward += 1.5
        else:
            reward -= 2.0

        return reward

    def is_success(self, physics, constraints):
        return (
            super().is_success(physics, constraints)
            and physics["efficiency"] >= self.min_efficiency
        )


TASKS = {
    "feasibility": lambda: FeasibilityTask(),
    "target_pr": lambda: TargetPRTask(),
    "target_pr_efficiency": lambda: TargetPREfficiencyTask(),
}


def get_task(task_name):
    try:
        return TASKS[task_name]()
    except KeyError as exc:
        raise KeyError(f"Unknown task: {task_name}") from exc
