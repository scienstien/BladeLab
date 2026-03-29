"""Reward calculation module for TurboDesigner 2.0"""


class Reward:
    """Handles reward calculations for optimization."""

    def __init__(self):
        pass

    def calculate_reward(self, state, action, next_state):
        """Calculate reward based on state transition."""
        raise NotImplementedError("Subclasses must implement this method")

    def sparse_reward(self, goal_reached):
        """Return sparse reward based on goal completion."""
        return 1.0 if goal_reached else 0.0

    def dense_reward(self, distance_to_goal):
        """Return dense reward based on distance to goal."""
        return 1.0 / (1.0 + distance_to_goal)
