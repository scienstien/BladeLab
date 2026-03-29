"""Task definitions module for TurboDesigner 2.0"""


class Task:
    """Base class for task definitions."""

    def __init__(self, name, description=""):
        self.name = name
        self.description = description

    def reset(self):
        """Reset task to initial state."""
        raise NotImplementedError("Subclasses must implement this method")

    def step(self, action):
        """Execute one step of the task."""
        raise NotImplementedError("Subclasses must implement this method")

    def is_complete(self):
        """Check if task is complete."""
        raise NotImplementedError("Subclasses must implement this method")


class TaskRegistry:
    """Registry for managing available tasks."""

    def __init__(self):
        self._tasks = {}

    def register(self, name, task_class):
        """Register a task class."""
        self._tasks[name] = task_class

    def get(self, name):
        """Get a task class by name."""
        return self._tasks.get(name)
