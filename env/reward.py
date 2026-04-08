"""Shared normalized scoring helpers for concrete tasks and graders."""


def clamp_score(value):
    return max(0.0, min(1.0, value))


def _feasibility_margin_score(constraints):
    margin_bonus = min(constraints["surge_margin"], constraints["choke_margin"])
    return clamp_score(0.75 + 0.25 * max(0.0, margin_bonus))


def _target_pr_score(physics, task_config):
    pr_error = abs(physics["pressure_ratio"] - task_config.target_pr)
    normalized_error = pr_error / max(task_config.target_pr, 1e-6)
    return clamp_score(1.0 - normalized_error)


def compute_score(physics, constraints, task_config):
    if not constraints["feasible"]:
        return 0.0

    task_name = getattr(task_config, "name", "")
    if task_name == "feasibility":
        return _feasibility_margin_score(constraints)

    pr_score = _target_pr_score(physics, task_config)
    if task_name == "target_pr":
        return pr_score

    if task_name == "target_pr_efficiency":
        efficiency_score = clamp_score(physics["efficiency"] / max(task_config.min_efficiency, 1e-6))
        return clamp_score(0.6 * pr_score + 0.4 * efficiency_score)

    return clamp_score(pr_score)
