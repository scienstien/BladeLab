"""Concrete optimization tasks for the environment."""

from env.config import CHOKE_LIMIT, MAX_STEPS, PR_TARGET, SURGE_LIMIT
from env.graders import grade_efficiency, grade_feasibility, grade_target_pr
from env.reward import compute_score


class Task:
    def __init__(self, name, description="", max_steps=MAX_STEPS, reward_scale=10.0, grader=None):
        self.name = name
        self.description = description
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.grader = grader

    def reset(self):
        return None

    def compute_reward(self, physics, constraints, prev_physics=None):
        raise NotImplementedError

    def score(self, physics, constraints):
        return compute_score(physics, constraints, self)

    def is_success(self, physics, constraints):
        raise NotImplementedError

    def is_done(self, step_count, physics, constraints):
        return step_count >= self.max_steps or self.is_success(physics, constraints)


class FeasibilityTask(Task):
    grader = grade_feasibility

    def __init__(self, surge_limit=SURGE_LIMIT, choke_limit=CHOKE_LIMIT, max_steps=MAX_STEPS):
        super().__init__(
            name="feasibility",
            description="Keep the design within surge and choke bounds.",
            max_steps=max_steps,
            grader=grade_feasibility,
        )
        self.surge_limit = surge_limit
        self.choke_limit = choke_limit

    def compute_reward(self, physics, constraints, prev_physics=None):
        return self.reward_scale * self.score(physics, constraints)

    def is_success(self, physics, constraints):
        return constraints["feasible"]


class TargetPRTask(Task):
    grader = grade_target_pr

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
            grader=grade_target_pr,
        )
        self.target_pr = target_pr
        self.pr_tolerance = pr_tolerance
        self.surge_limit = surge_limit
        self.choke_limit = choke_limit

    def compute_reward(self, physics, constraints, prev_physics=None):
        return self.reward_scale * self.score(physics, constraints)

    def is_success(self, physics, constraints):
        return constraints["feasible"] and abs(physics["pressure_ratio"] - self.target_pr) <= self.pr_tolerance


class TargetPREfficiencyTask(TargetPRTask):
    grader = grade_efficiency

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
        self.grader = grade_efficiency

    def compute_reward(self, physics, constraints, prev_physics=None):
        return self.reward_scale * self.score(physics, constraints)

    def is_success(self, physics, constraints):
        return (
            super().is_success(physics, constraints)
            and physics["efficiency"] >= self.min_efficiency
        )


TASKS = {
    "feasibility": FeasibilityTask,
    "target_pr": TargetPRTask,
    "target_pr_efficiency": TargetPREfficiencyTask,
}
def get_task(task_name):
    try:
        return TASKS[task_name]()
    except KeyError as exc:
        raise KeyError(f"Unknown task: {task_name}") from exc
