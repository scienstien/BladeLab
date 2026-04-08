---
title: "Fix rollout success logic using wrong score for all tasks"
priority: HIGH
labels: bug, high-priority
file: api/routes.py:232
---

## Problem Description
The rollout endpoint uses `feasible_score` for ALL tasks instead of task-specific criteria. The `target_pr` task should use `pr_score` and the `target_pr_efficiency` task should use `efficiency_score` for success determination.

## Current Code
```python
response = {
    "trajectory": result["trajectory"],
    "total_reward": result["total_reward"],
    "final_state": result["final_state"],
    "success": result["feasible_score"] == 1.0,  # BUG: Uses feasible_score for ALL tasks
    "steps": len(result["trajectory"]),
    "scores": {
        "feasibility_score": result["feasible_score"],
        "pr_score": result["pr_score"],
        "efficiency_score": result["efficiency_score"]
    }
}
```

## Impact
- `target_pr` task reports success even when PR target is not met (only checks feasibility)
- `target_pr_efficiency` task reports success even when efficiency target is not met
- API clients receive incorrect success/failure determination
- Evaluation metrics are misleading

## Suggested Fix
Use task-specific success criteria:
```python
if req.task_name == "feasibility":
    success = result["feasible_score"] == 1.0
elif req.task_name == "target_pr":
    success = result["pr_score"] == 1.0
elif req.task_name == "target_pr_efficiency":
    success = result["efficiency_score"] == 1.0
```
