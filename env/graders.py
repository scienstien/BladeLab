"""Task-aware grading helpers for TurboDesigner 2.0."""

from env.config import PR_TARGET
from env.tasks import FeasibilityTask, TargetPRTask, TargetPREfficiencyTask


def clamp_score(value):
    return max(0.0, min(1.0, value))


class Grader:
    """Base class for grading task performance."""

    def grade(self, physics, constraints):
        raise NotImplementedError("Subclasses must implement this method")

    def passed(self, physics, constraints):
        raise NotImplementedError("Subclasses must implement this method")


class PassFailGrader(Grader):
    """Simple pass/fail grader for scalar scores."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def grade(self, score, constraints=None):
        return 1.0 if score >= self.threshold else 0.0

    def passed(self, score, constraints=None):
        return score >= self.threshold


class FeasibilityGrader(Grader):
    def __init__(self, task=None):
        self.task = task or FeasibilityTask()

    def grade(self, physics, constraints):
        if constraints["feasible"]:
            margin_bonus = min(constraints["surge_margin"], constraints["choke_margin"])
            return clamp_score(0.75 + 0.25 * max(0.0, margin_bonus))
        return 0.0

    def passed(self, physics, constraints):
        return self.task.is_success(physics, constraints)


class TargetPRGrader(Grader):
    def __init__(self, task=None):
        self.task = task or TargetPRTask()

    def grade(self, physics, constraints):
        if not constraints["feasible"]:
            return 0.0

        pr_error = abs(physics["pressure_ratio"] - self.task.target_pr)
        normalized_error = pr_error / max(self.task.target_pr, 1e-6)
        return clamp_score(1.0 - normalized_error)

    def passed(self, physics, constraints):
        return self.task.is_success(physics, constraints)


class TargetPREfficiencyGrader(Grader):
    def __init__(self, task=None):
        self.task = task or TargetPREfficiencyTask()
        self.pr_grader = TargetPRGrader(task=self.task)

    def grade(self, physics, constraints):
        if not constraints["feasible"]:
            return 0.0

        pr_score = self.pr_grader.grade(physics, constraints)
        efficiency_score = clamp_score(physics["efficiency"] / max(self.task.min_efficiency, 1e-6))
        return clamp_score(0.6 * pr_score + 0.4 * efficiency_score)

    def passed(self, physics, constraints):
        return self.task.is_success(physics, constraints)


def grade_feasibility(physics, constraints, task=None):
    return FeasibilityGrader(task=task).grade(physics, constraints)


def grade_target_pr(physics, constraints, task=None):
    return TargetPRGrader(task=task or TargetPRTask(target_pr=PR_TARGET)).grade(physics, constraints)


def grade_efficiency(physics, constraints, task=None):
    return TargetPREfficiencyGrader(
        task=task or TargetPREfficiencyTask(target_pr=PR_TARGET)
    ).grade(physics, constraints)
