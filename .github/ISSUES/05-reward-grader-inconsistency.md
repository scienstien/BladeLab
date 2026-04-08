---
title: "Resolve inconsistency between task reward functions and graders"
priority: MEDIUM
labels: bug, medium-priority
files:
  - env/tasks.py
  - env/graders.py
---

## Problem Description
The `TargetPRTask` uses a quadratic penalty in its reward function (`quadratic_target_penalty`), while the `TargetPRGrader` uses a linear normalized error for grading. This creates inconsistency where the same objective is scored differently during training vs evaluation.

## Current Code
**Task (quadratic penalty):** `env/tasks.py:68-81`
```python
def compute_reward(self, physics, constraints, prev_physics=None):
    pr_error = abs(physics["pressure_ratio"] - self.target_pr)
    reward = -0.01
    if not constraints["feasible"]:
        reward -= 12.0 + infeasibility_penalty(constraints, margin_weight=25.0)
    else:
        reward += 3.0
    reward -= quadratic_target_penalty(physics["pressure_ratio"], self.target_pr, weight=8.0)  # Quadratic
    if pr_error <= self.pr_tolerance and constraints["feasible"]:
        reward += 2.0
    return reward
```

**Grader (linear penalty):** `env/graders.py:48-61`
```python
class TargetPRGrader(Grader):
    def grade(self, physics, constraints):
        if not constraints["feasible"]:
            return 0.0
        pr_error = abs(physics["pressure_ratio"] - self.task.target_pr)
        normalized_error = pr_error / max(self.task.target_pr, 1e-6)
        return clamp_score(1.0 - normalized_error)  # Linear
```

## Impact
- RL agent optimizes for quadratic error during training
- Evaluation uses linear error, causing misalignment
- Agent may appear to perform differently in evaluation vs training

## Suggested Fix
Align the reward and grading approaches - either use quadratic in both, linear in both, or document the intentional difference.
