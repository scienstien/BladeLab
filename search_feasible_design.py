import math
import random
from copy import deepcopy

from env.config import BOUNDS, INIT_PARAMS
from env.constraints import check_constraints
from env.physics import compute_physics


SEARCH_SEED = 7
NUM_SAMPLES = 4000
TOP_K = 20
REFINE_STEPS = 120


def clamp_state(state):
    clamped = dict(state)
    for key, (low, high) in BOUNDS.items():
        clamped[key] = max(low, min(high, clamped[key]))
    clamped["Z"] = int(round(clamped["Z"]))
    return clamped


def evaluate(state):
    physics = compute_physics(state)
    constraints = check_constraints(physics)
    pr = physics["pressure_ratio"]
    eff = physics["efficiency"]

    penalty = 0.0
    if constraints["surge"]:
        penalty += 1000.0 + 1000.0 * abs(constraints["surge_margin"])
    if constraints["choke"]:
        penalty += 1000.0 + 1000.0 * abs(constraints["choke_margin"])

    objective = penalty
    objective += abs(pr - 1.2) * 20.0
    objective -= eff * 10.0

    return {
        "state": dict(state),
        "physics": physics,
        "constraints": constraints,
        "objective": objective,
    }


def random_state():
    state = deepcopy(INIT_PARAMS)
    state["r2"] = random.uniform(*BOUNDS["r2"])
    state["blade_angle"] = random.uniform(*BOUNDS["blade_angle"])
    state["b2"] = random.uniform(*BOUNDS["b2"])
    state["Z"] = random.randint(*BOUNDS["Z"])
    return clamp_state(state)


def perturb_state(state, scale):
    candidate = dict(state)
    candidate["r2"] += random.uniform(-scale, scale) * 0.02
    candidate["blade_angle"] += random.uniform(-scale, scale) * 6.0
    candidate["b2"] += random.uniform(-scale, scale) * 0.004
    candidate["Z"] += random.choice([-1, 0, 1])
    return clamp_state(candidate)


def summarize(result):
    return {
        "state": result["state"],
        "pressure_ratio": result["physics"]["pressure_ratio"],
        "efficiency": result["physics"]["efficiency"],
        "mass_flow": result["physics"]["mass_flow"],
        "feasible": result["constraints"]["feasible"],
        "surge_margin": result["constraints"]["surge_margin"],
        "choke_margin": result["constraints"]["choke_margin"],
        "choke_limit": result["constraints"]["choke_limit"],
        "objective": result["objective"],
    }


def main():
    random.seed(SEARCH_SEED)

    population = [evaluate(random_state()) for _ in range(NUM_SAMPLES)]
    population.sort(key=lambda item: item["objective"])
    elite = population[:TOP_K]

    best = elite[0]

    for step in range(REFINE_STEPS):
        scale = max(0.15, 1.0 - step / REFINE_STEPS)
        proposals = []
        for parent in elite:
            proposals.append(evaluate(perturb_state(parent["state"], scale)))
        proposals.sort(key=lambda item: item["objective"])
        merged = sorted(elite + proposals, key=lambda item: item["objective"])
        elite = merged[:TOP_K]
        if elite[0]["objective"] < best["objective"]:
            best = elite[0]

    print("BEST")
    print(summarize(best))
    print()
    print("TOP_FEASIBLE")
    feasible = [item for item in sorted(population + elite, key=lambda item: item["objective"]) if item["constraints"]["feasible"]]
    for result in feasible[:10]:
        print(summarize(result))


if __name__ == "__main__":
    main()
