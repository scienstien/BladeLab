"""Grading module for TurboDesigner 2.0"""

from env.config import PR_TARGET


class Grader:
    """Base class for grading task performance."""

    def __init__(self):
        pass

    def grade(self, result):
        """Grade a task result."""
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_score(self, metrics):
        """Calculate overall score from metrics."""
        raise NotImplementedError("Subclasses must implement this method")


class PassFailGrader(Grader):
    """Simple pass/fail grader."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def grade(self, result):
        """Return pass/fail based on threshold."""
        return result >= self.threshold

    def calculate_score(self, metrics):
        """Calculate binary score."""
        return 1.0 if self.grade(metrics) else 0.0


def grade_feasibility(physics, constraints):
    """Grade feasibility - pass if no constraints violated."""
    if constraints["surge"] or constraints["choke"]:
        return 0.0
    return 1.0


def grade_target_pr(physics, constraints):
    """Grade target pressure ratio achievement."""
    pr = physics["pressure_ratio"]
    if pr >= PR_TARGET:
        return 1.0
    return max(0.0, pr / PR_TARGET)


def grade_efficiency(physics, constraints):
    """Grade efficiency performance."""
    eff = physics["efficiency"]
    return min(1.0, eff / 0.8)
