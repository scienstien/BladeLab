"""Tests for environment module"""

import pytest
from env.physics import Physics
from env.reward import Reward
from env.tasks import Task, TaskRegistry
from env.graders import Grader, PassFailGrader


class TestPhysics:
    """Test physics calculations."""

    def test_calculate_force(self):
        """Test force calculation."""
        physics = Physics()
        assert physics.calculate_force(10, 5) == 50

    def test_calculate_velocity(self):
        """Test velocity calculation."""
        physics = Physics()
        assert physics.calculate_velocity(0, 10, 5) == 50


class TestReward:
    """Test reward calculations."""

    def test_sparse_reward(self):
        """Test sparse reward."""
        reward = Reward()
        assert reward.sparse_reward(True) == 1.0
        assert reward.sparse_reward(False) == 0.0

    def test_dense_reward(self):
        """Test dense reward."""
        reward = Reward()
        assert reward.dense_reward(0) == 1.0
        assert reward.dense_reward(1) == 0.5


class TestTask:
    """Test task functionality."""

    def test_task_creation(self):
        """Test task creation."""
        task = Task("test_task", "A test task")
        assert task.name == "test_task"
        assert task.description == "A test task"


class TestGraders:
    """Test grading functionality."""

    def test_pass_fail_grader(self):
        """Test pass/fail grader."""
        grader = PassFailGrader(threshold=0.5)
        assert grader.grade(0.6) is True
        assert grader.grade(0.4) is False
