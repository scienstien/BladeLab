from copy import deepcopy

from env.config import CHOKE_LIMIT, INIT_PARAMS, PR_TARGET, SURGE_LIMIT
from env.constraints import check_constraints
from env.core_env import BladeLabEnv, apply_action
from env.physics import compute_physics
from env.tasks import FeasibilityTask, TargetPRTask, TargetPREfficiencyTask


def test_apply_action_clamps_state():
    state = deepcopy(INIT_PARAMS)
    updated = apply_action(
        state,
        {
            "delta_r2": 10.0,
            "delta_angle": 100.0,
            "delta_b2": -10.0,
            "delta_Z": 100.0,
        },
    )

    assert updated["r2"] == 0.2
    assert updated["blade_angle"] == 80
    assert updated["b2"] == 0.005
    assert updated["Z"] == 20


def test_compute_physics_is_pure_and_derives_geometry():
    state = deepcopy(INIT_PARAMS)
    physics = compute_physics(state)

    assert state == INIT_PARAMS
    assert physics["D1"] == 2.0 * state["r1"]
    assert physics["D2"] == 2.0 * state["r2"]
    assert physics["mass_flow"] > 0.0
    assert physics["losses"] == sum(physics["loss_breakdown"].values())


def test_constraints_detect_feasible_and_infeasible_regions():
    feasible_constraints = check_constraints({"mass_flow": 0.9})
    surge_constraints = check_constraints({"mass_flow": SURGE_LIMIT - 0.01})
    choke_constraints = check_constraints({"mass_flow": CHOKE_LIMIT + 0.01})

    assert feasible_constraints["feasible"] is True
    assert surge_constraints["surge"] is True
    assert choke_constraints["choke"] is True


def test_task_reward_ordering_prefers_feasible_targeted_design():
    task = TargetPREfficiencyTask(target_pr=PR_TARGET, pr_tolerance=0.05, min_efficiency=0.75)

    good_physics = {"pressure_ratio": PR_TARGET, "efficiency": 0.8}
    good_constraints = {
        "feasible": True,
        "surge": False,
        "choke": False,
        "surge_margin": 0.1,
        "choke_margin": 0.1,
    }

    bad_physics = {"pressure_ratio": PR_TARGET + 0.4, "efficiency": 0.85}
    bad_constraints = {
        "feasible": False,
        "surge": True,
        "choke": False,
        "surge_margin": -0.2,
        "choke_margin": 0.3,
    }

    assert task.compute_reward(good_physics, good_constraints) > task.compute_reward(
        bad_physics, bad_constraints
    )


def test_concrete_tasks_report_success_consistently():
    feasible_task = FeasibilityTask()
    pr_task = TargetPRTask(target_pr=PR_TARGET, pr_tolerance=0.05)
    hard_task = TargetPREfficiencyTask(target_pr=PR_TARGET, pr_tolerance=0.05, min_efficiency=0.75)

    physics = {"pressure_ratio": PR_TARGET + 0.01, "efficiency": 0.8}
    constraints = {
        "feasible": True,
        "surge": False,
        "choke": False,
        "surge_margin": 0.1,
        "choke_margin": 0.1,
    }

    assert feasible_task.is_success(physics, constraints) is True
    assert pr_task.is_success(physics, constraints) is True
    assert hard_task.is_success(physics, constraints) is True


def test_env_uses_selected_task():
    env = BladeLabEnv(task_name="target_pr_efficiency")
    obs = env.reset()

    assert obs["feasible"] in (True, False)
    _, _, _, info = env.step({})
    assert info["task"] == "target_pr_efficiency"
