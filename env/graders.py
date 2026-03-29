"""Grading module for TurboDesigner 2.0"""


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
