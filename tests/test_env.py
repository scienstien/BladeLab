from copy import deepcopy

from env.config import CHOKE_LIMIT, INIT_PARAMS, PR_TARGET, SURGE_LIMIT
from env.constraints import check_constraints
from env.core_env import BladeLabEnv, apply_action
from env.graders import FeasibilityGrader, TargetPRGrader, TargetPREfficiencyGrader
from env.models import Action, Observation, StepInfo, safe_default_action
from env.physics import compute_physics
from env.tasks import FeasibilityTask, TargetPRTask, TargetPREfficiencyTask, TASKS


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

    assert updated == INIT_PARAMS


def test_compute_physics_is_pure_and_derives_geometry():
    state = deepcopy(INIT_PARAMS)
    physics = compute_physics(state)

    assert state == INIT_PARAMS
    assert physics["D1"] == 2.0 * state["r1"]
    assert physics["D2"] == 2.0 * state["r2"]
    assert physics["mass_flow"] > 0.0
    assert physics["losses"] == sum(physics["loss_breakdown"].values())


def test_constraints_detect_feasible_and_infeasible_regions():
    base_physics = compute_physics(deepcopy(INIT_PARAMS))
    feasible_case = dict(base_physics)
    feasible_case["mass_flow"] = min(base_physics["mass_flow"], check_constraints(base_physics)["choke_limit"] - 0.05)
    surge_case = dict(base_physics)
    surge_case["mass_flow"] = SURGE_LIMIT - 0.01
    choke_case = dict(base_physics)
    choke_case["mass_flow"] = check_constraints(base_physics)["choke_limit"] + 0.01

    feasible_constraints = check_constraints(feasible_case)
    surge_constraints = check_constraints(surge_case)
    choke_constraints = check_constraints(choke_case)

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

    assert isinstance(obs, Observation)
    assert obs.feasible in (True, False)
    _, _, _, info = env.step({})
    assert isinstance(info, StepInfo)
    assert info.task == "target_pr_efficiency"


def test_graders_follow_task_success_criteria():
    physics = {"pressure_ratio": PR_TARGET + 0.01, "efficiency": 0.8}
    constraints = {
        "feasible": True,
        "surge": False,
        "choke": False,
        "surge_margin": 0.1,
        "choke_margin": 0.1,
    }

    feasibility_grader = FeasibilityGrader(task=FeasibilityTask())
    pr_grader = TargetPRGrader(task=TargetPRTask(target_pr=PR_TARGET, pr_tolerance=0.05))
    hard_grader = TargetPREfficiencyGrader(
        task=TargetPREfficiencyTask(target_pr=PR_TARGET, pr_tolerance=0.05, min_efficiency=0.75)
    )

    assert feasibility_grader.passed(physics, constraints) is True
    assert pr_grader.passed(physics, constraints) is True
    assert hard_grader.passed(physics, constraints) is True
    assert 0.0 <= feasibility_grader.grade(physics, constraints) <= 1.0
    assert 0.0 <= pr_grader.grade(physics, constraints) <= 1.0
    assert 0.0 <= hard_grader.grade(physics, constraints) <= 1.0


def test_infeasible_design_gets_zero_task_grades():
    physics = {"pressure_ratio": PR_TARGET, "efficiency": 0.9}
    constraints = {
        "feasible": False,
        "surge": True,
        "choke": False,
        "surge_margin": -0.2,
        "choke_margin": 0.3,
    }

    assert FeasibilityGrader().grade(physics, constraints) == 0.0
    assert TargetPRGrader().grade(physics, constraints) == 0.0
    assert TargetPREfficiencyGrader().grade(physics, constraints) == 0.0


def test_action_model_validates_and_safe_default_is_zero():
    action = Action(delta_r2=0.001, delta_angle=1.0, delta_b2=0.0005, delta_Z=1)
    default_action = safe_default_action()

    assert action.delta_r2 == 0.001
    assert default_action.delta_r2 == 0.0
    assert default_action.delta_angle == 0.0
    assert default_action.delta_b2 == 0.0
    assert default_action.delta_Z == 0


def test_env_step_accepts_invalid_action_and_falls_back_safely():
    env = BladeLabEnv(task_name="feasibility")
    env.reset()
    obs, reward, done, info = env.step({"delta_r2": 999, "bad_key": 1})

    assert isinstance(obs, Observation)
    assert isinstance(info, StepInfo)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_env_state_returns_typed_observation():
    env = BladeLabEnv(task_name="feasibility")
    env.reset()

    assert isinstance(env.state(), Observation)


def test_task_registry_contains_three_tasks():
    assert set(TASKS.keys()) == {"feasibility", "target_pr", "target_pr_efficiency"}
