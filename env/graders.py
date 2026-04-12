"""Task-aware grading helpers for TurboDesigner 2.0."""

from types import SimpleNamespace

from env.config import PR_TARGET
from env.reward import compute_score


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
        self.task = task or SimpleNamespace(name="feasibility")

    def grade(self, physics, constraints):
        return compute_score(physics, constraints, self.task)

    def passed(self, physics, constraints):
        return constraints["feasible"]


class TargetPRGrader(Grader):
    def __init__(self, task=None, target_pr=PR_TARGET, pr_tolerance=0.05):
        self.task = task or SimpleNamespace(
            name="target_pr",
            target_pr=target_pr,
            pr_tolerance=pr_tolerance,
        )

    def grade(self, physics, constraints):
        return compute_score(physics, constraints, self.task)

    def passed(self, physics, constraints):
        return (
            constraints["feasible"]
            and abs(physics["pressure_ratio"] - self.task.target_pr) <= self.task.pr_tolerance
        )


class TargetPREfficiencyGrader(Grader):
    def __init__(
        self,
        task=None,
        target_pr=PR_TARGET,
        pr_tolerance=0.05,
        min_efficiency=0.75,
    ):
        self.task = task or SimpleNamespace(
            name="target_pr_efficiency",
            target_pr=target_pr,
            pr_tolerance=pr_tolerance,
            min_efficiency=min_efficiency,
        )

    def grade(self, physics, constraints):
        return compute_score(physics, constraints, self.task)

    def passed(self, physics, constraints):
        return (
            constraints["feasible"]
            and abs(physics["pressure_ratio"] - self.task.target_pr) <= self.task.pr_tolerance
            and physics["efficiency"] >= self.task.min_efficiency
        )


def grade_feasibility(physics, constraints, task=None):
    return FeasibilityGrader(task=task).grade(physics, constraints)


def grade_target_pr(physics, constraints, task=None):
    return TargetPRGrader(task=task).grade(physics, constraints)


def grade_efficiency(physics, constraints, task=None):
    return TargetPREfficiencyGrader(task=task).grade(physics, constraints)
