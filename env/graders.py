"""Task-aware grading helpers for TurboDesigner 2.0."""

from env.config import PR_TARGET
from env.reward import compute_score
from env.tasks import FeasibilityTask, TargetPRTask, TargetPREfficiencyTask


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
        return compute_score(physics, constraints, self.task)

    def passed(self, physics, constraints):
        return self.task.is_success(physics, constraints)


class TargetPRGrader(Grader):
    def __init__(self, task=None):
        self.task = task or TargetPRTask()

    def grade(self, physics, constraints):
        return compute_score(physics, constraints, self.task)

    def passed(self, physics, constraints):
        return self.task.is_success(physics, constraints)


class TargetPREfficiencyGrader(Grader):
    def __init__(self, task=None):
        self.task = task or TargetPREfficiencyTask()

    def grade(self, physics, constraints):
        return compute_score(physics, constraints, self.task)

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
