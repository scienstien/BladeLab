"""Reward helpers shared by concrete tasks."""


def efficiency_bonus(physics, weight=1.0):
    return weight * physics["efficiency"]


def quadratic_target_penalty(value, target, weight=1.0):
    return weight * (value - target) ** 2


def infeasibility_penalty(constraints, base_penalty=0.0, margin_weight=1.0):
    penalty = base_penalty
    if constraints["surge"]:
        penalty += margin_weight * abs(constraints["surge_margin"])
    if constraints["choke"]:
        penalty += margin_weight * abs(constraints["choke_margin"])
    return penalty
