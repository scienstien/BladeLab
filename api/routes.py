"""API routes for TurboDesigner 2.0"""

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from api.schemas import EvaluateRequest, PredictRequest, RolloutRequest
from env.core_env import BladeLabEnv
from env.tasks import TASKS
from inference import (
    Agent,
    HeuristicPolicy,
    evaluate_agent,
    load_openai_policy,
    run_episode,
)

bp = Blueprint("api", __name__, url_prefix="/api")

# Valid task names
VALID_TASKS = list(TASKS.keys())

# Valid policy types
VALID_POLICY_TYPES = ["heuristic", "openai"]

# Default max steps
DEFAULT_MAX_STEPS = 100


def _validation_response(details):
    return jsonify({"error": "Validation failed", "details": details}), 400


def _validation_details_from_exception(exc):
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "request"
        message = error.get("msg", "Invalid value")
        details.append(f"{location}: {message}")
    return details or ["Invalid request payload"]


def _json_payload():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None
    return payload


def _is_positive_int(value):
    return isinstance(value, int) and value > 0


@bp.errorhandler(ValidationError)
def handle_validation_error(exc):
    return _validation_response(_validation_details_from_exception(exc))


@bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction request with policy-based inference.

    Expects JSON body with:
    - observation: list or dict (required) - The observation/state input
    - policy_type: str (optional) - Either "heuristic" or "openai" (default: "heuristic")

    Returns:
    - action: the predicted action
    - policy_type: the policy used for prediction
    """
    payload = _json_payload()
    if payload is None:
        return _validation_response(["Request body must be a JSON object"])

    req = PredictRequest.model_validate(payload)

    details = []
    if req.task_name not in VALID_TASKS:
        details.append(
            f"task_name: invalid task '{req.task_name}'. Valid tasks: {VALID_TASKS}"
        )
    if req.policy_type not in VALID_POLICY_TYPES:
        details.append(
            f"policy_type: invalid policy '{req.policy_type}'. Valid policy types: {VALID_POLICY_TYPES}"
        )
    if req.policy_type == "openai" and not req.model_name:
        details.append("model_name: required when policy_type is 'openai'")

    if details:
        return _validation_response(details)

    try:
        if req.policy_type == "heuristic":
            policy = HeuristicPolicy()
        else:
            policy = load_openai_policy(task_name=req.task_name, model_name=req.model_name)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load policy: {str(e)}"}), 500

    # Create agent and get action
    agent = Agent(policy)
    action = agent.act(req.observation)

    observation = (
        req.observation.model_dump()
        if hasattr(req.observation, "model_dump")
        else req.observation
    )

    # Return structured response
    return jsonify({
        "action": action.model_dump(),
        "policy_type": req.policy_type,
        "observation": observation
    })


@bp.route("/tasks", methods=["GET"])
def get_tasks():
    """Get available tasks with descriptions."""
    tasks_info = {
        "feasibility": {
            "name": "FeasibilityTask",
            "description": "Keep design within surge/choke bounds",
            "difficulty": "easy",
            "success_criteria": "constraints['feasible'] == True"
        },
        "target_pr": {
            "name": "TargetPRTask",
            "description": "Match target pressure ratio (2.5) within 5% tolerance",
            "difficulty": "medium",
            "success_criteria": "feasible AND abs(PR - 2.5) <= 0.05"
        },
        "target_pr_efficiency": {
            "name": "TargetPREfficiencyTask",
            "description": "Match PR AND maximize efficiency (min 75%)",
            "difficulty": "hard",
            "success_criteria": "target_pr success AND efficiency >= 0.75"
        }
    }
    return jsonify({"tasks": tasks_info})


@bp.route("/rollout", methods=["POST"])
def rollout():
    """
    Run a rollout episode with the specified policy and task.

    Expects JSON body with:
    - task_name: str (required) - Name of the task from TASKS registry
    - policy_type: str (required) - Either "heuristic" or "openai"
    - max_steps: int (optional) - Maximum steps for the episode (default: 100)
    - model_name: str (optional) - OpenAI model name (only used for policy_type="openai")

    Returns:
    - trajectory: list of step dicts with state, action, reward, next_state, info
    - total_reward: float - Sum of all rewards
    - final_state: dict - Final observation state
    - success: bool - Whether the task was completed successfully
    - steps: int - Number of steps taken
    - scores: dict with feasibility_score, pr_score, efficiency_score
    """
    payload = _json_payload()
    if payload is None:
        return _validation_response(["Request body must be a JSON object"])

    req = RolloutRequest.model_validate(payload)

    details = []
    if "task_name" not in payload:
        details.append("task_name: field is required")
    if "policy_type" not in payload:
        details.append("policy_type: field is required")
    if req.task_name not in VALID_TASKS:
        details.append(
            f"task_name: invalid task '{req.task_name}'. Valid tasks: {VALID_TASKS}"
        )
    if req.policy_type not in VALID_POLICY_TYPES:
        details.append(
            f"policy_type: invalid policy '{req.policy_type}'. Valid policy types: {VALID_POLICY_TYPES}"
        )

    max_steps = req.max_steps if "max_steps" in payload else DEFAULT_MAX_STEPS
    if not _is_positive_int(max_steps):
        details.append("max_steps: must be a positive integer")
    if req.policy_type == "openai" and not req.model_name:
        details.append("model_name: required when policy_type is 'openai'")

    if details:
        return _validation_response(details)

    # Load the appropriate policy
    try:
        if req.policy_type == "heuristic":
            policy = HeuristicPolicy()
        elif req.policy_type == "openai":
            policy = load_openai_policy(task_name=req.task_name, model_name=req.model_name)
        else:
            # This should not happen due to validation above
            return _validation_response([f"policy_type: unsupported policy '{req.policy_type}'"])
    except RuntimeError as e:
        # Handle missing API key or other runtime errors
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load policy: {str(e)}"}), 500

    # Create agent with the loaded policy
    agent = Agent(policy)

    # Create fresh environment with the specified task
    env = BladeLabEnv(task_name=req.task_name)

    # Run the episode
    try:
        result = run_episode(env, agent, max_steps=max_steps)
    except Exception as e:
        return jsonify({"error": f"Episode execution failed: {str(e)}"}), 500

    # Build structured response
    response = {
        "trajectory": result["trajectory"],
        "total_reward": result["total_reward"],
        "final_state": result["final_state"],
        "success": result["feasible_score"] == 1.0,  # Success based on feasibility
        "steps": len(result["trajectory"]),
        "scores": {
            "feasibility_score": result["feasible_score"],
            "pr_score": result["pr_score"],
            "efficiency_score": result["efficiency_score"]
        }
    }

    return jsonify(response)


@bp.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Evaluate a policy over multiple episodes and return statistical summary.

    Expects JSON body with:
    - task_name: str (required) - Name of the task from TASKS registry
    - policy_type: str (required) - Either "heuristic" or "openai"
    - num_episodes: int (optional) - Number of episodes to run (default: 10)
    - max_steps: int (optional) - Maximum steps per episode (default: None)
    - model_name: str (optional) - OpenAI model name (only used for policy_type="openai")

    Returns:
    - task_name: str - The task that was evaluated
    - policy_type: str - The policy type used
    - num_episodes: int - Number of episodes run
    - reward_mean: float - Mean total reward across episodes
    - reward_variance: float - Variance of total rewards
    - pr_mean: float - Mean final pressure ratio
    - pr_variance: float - Variance of pressure ratios
    - efficiency_mean: float - Mean final efficiency
    - efficiency_variance: float - Variance of efficiencies
    - mass_flow_mean: float - Mean final mass flow
    - mass_flow_variance: float - Variance of mass flows
    """
    payload = _json_payload()
    if payload is None:
        return _validation_response(["Request body must be a JSON object"])

    req = EvaluateRequest.model_validate(payload)

    details = []
    if "task_name" not in payload:
        details.append("task_name: field is required")
    if req.task_name not in VALID_TASKS:
        details.append(
            f"task_name: invalid task '{req.task_name}'. Valid tasks: {VALID_TASKS}"
        )
    if req.policy_type not in VALID_POLICY_TYPES:
        details.append(
            f"policy_type: invalid policy '{req.policy_type}'. Valid policy types: {VALID_POLICY_TYPES}"
        )
    if not _is_positive_int(req.num_episodes):
        details.append("num_episodes: must be a positive integer")
    if req.max_steps is not None and not _is_positive_int(req.max_steps):
        details.append("max_steps: must be a positive integer or null")
    if req.policy_type == "openai" and not req.model_name:
        details.append("model_name: required when policy_type is 'openai'")

    if details:
        return _validation_response(details)

    # Load the appropriate policy
    try:
        if req.policy_type == "heuristic":
            policy = HeuristicPolicy()
        elif req.policy_type == "openai":
            policy = load_openai_policy(task_name=req.task_name, model_name=req.model_name)
        else:
            return _validation_response([f"policy_type: unsupported policy '{req.policy_type}'"])
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load policy: {str(e)}"}), 500

    # Create agent with the loaded policy
    agent = Agent(policy)

    # Run evaluation over multiple episodes
    try:
        summary = evaluate_agent(
            agent=agent,
            task_name=req.task_name,
            num_episodes=req.num_episodes,
            max_steps=req.max_steps,
        )
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

    # Build response with statistical summary
    response = {
        "task_name": req.task_name,
        "policy_type": req.policy_type,
        "num_episodes": req.num_episodes,
        "reward_mean": summary["reward_mean"],
        "reward_variance": summary["reward_variance"],
        "pr_mean": summary["pr_mean"],
        "pr_variance": summary["pr_variance"],
        "efficiency_mean": summary["efficiency_mean"],
        "efficiency_variance": summary["efficiency_variance"],
        "mass_flow_mean": summary["mass_flow_mean"],
        "mass_flow_variance": summary["mass_flow_variance"],
    }

    return jsonify(response)
