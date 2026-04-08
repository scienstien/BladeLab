import argparse
import json
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from env.core_env import BladeLabEnv
from env.graders import grade_efficiency, grade_feasibility, grade_target_pr
from env.models import Action, Observation, safe_default_action
from env.tasks import get_task


STATE_KEYS = [
    "efficiency",
    "pressure_ratio",
    "mass_flow",
    "feasible",
    "surge_margin",
    "choke_margin",
    "r2",
    "blade_angle",
    "b2",
    "Z",
]

ACTION_KEYS = ["delta_r2", "delta_angle", "delta_b2", "delta_Z"]

ACTION_SCALES = {
    "delta_r2": 0.006,
    "delta_angle": 4.0,
    "delta_b2": 0.002,
    "delta_Z": 1.0,
}



try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:
    class DeterministicActor(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            layers.append(nn.Tanh())
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)
else:
    class DeterministicActor:  # pragma: no cover - only used to raise a clear runtime error
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to load checkpoint-based policies.")


class TorchPolicy:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, state):
        state_tensor = torch.tensor(encode_state(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.model(state_tensor).squeeze(0).cpu().tolist()
        return validate_action(decode_action(action_tensor))


class HeuristicPolicy:
    """Deterministic fallback policy for debugging when no checkpoint is available."""

    def __call__(self, state):
        state = state.model_dump() if isinstance(state, Observation) else state
        action = {
            "delta_r2": 0.0,
            "delta_angle": 0.0,
            "delta_b2": 0.0,
            "delta_Z": 0,
        }

        if not state["feasible"]:
            if state["choke_margin"] < 0:
                action["delta_r2"] = -0.004
                action["delta_b2"] = -0.001
            elif state["surge_margin"] < 0:
                action["delta_r2"] = 0.004
                action["delta_b2"] = 0.001
            return action

        if state["pressure_ratio"] < 1.2:
            action["delta_angle"] = 2.0
            action["delta_Z"] = 1
        elif state["efficiency"] < 0.7:
            action["delta_r2"] = -0.001
            action["delta_b2"] = -0.0005

        return action


class Agent:
    def __init__(self, policy):
        self.policy = policy

    def act(self, observation, trajectory=None):
        if isinstance(self.policy, OpenAIPolicy):
            return validate_action(self.policy(observation, trajectory))
        return validate_action(self.policy(observation))

    def reset(self):
        pass


class OpenAIPolicy:
    def __init__(self, client, model, task_name):
        self.client = client
        self.model = model
        self.task_name = task_name

    def __call__(self, observation, trajectory=None):
        observation_payload = observation.model_dump() if isinstance(observation, Observation) else observation
        compact_trajectory = [
            {
                "reward": step["reward"],
                "action": step["action"],
                "next_state": step["next_state"],
            }
            for step in (trajectory or [])[-5:]
        ]

        prompt = {
            "task": self.task_name,
            "instruction": (
                "Return only a JSON object with keys delta_r2, delta_angle, delta_b2, delta_Z. "
                "Use conservative continuous action deltas to improve reward while keeping the design feasible."
            ),
            "observation": observation_payload,
            "recent_trajectory": compact_trajectory,
            "action_bounds": Action.model_json_schema(),
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a deterministic compressor-control policy. Output strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt),
                    },
                ],
                temperature=0,
                timeout=30,
            )
        except Exception as e:
            # Log API error and return safe default action
            error_type = type(e).__name__
            if "auth" in str(e).lower():
                print(f"[API_ERROR] Authentication failed: {e}")
            elif "rate" in str(e).lower():
                print(f"[API_ERROR] Rate limit exceeded: {e}")
            elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                print(f"[API_ERROR] Connection/timeout error: {e}")
            else:
                print(f"[API_ERROR] {error_type}: {e}")
            return safe_default_action()

        text = getattr(response, "output_text", "") or ""
        if not text:
            print("[API_ERROR] Empty response from API")
            return safe_default_action()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[API_ERROR] Invalid JSON response: {e}")
            print(f"[API_ERROR] Raw response (truncated): {text[:200]}")
            return safe_default_action()

        return validate_action(parsed)


def encode_state(state):
    state = state.model_dump() if isinstance(state, Observation) else state
    encoded = []
    for key in STATE_KEYS:
        value = state[key]
        if key == "feasible":
            encoded.append(1.0 if value else 0.0)
        else:
            encoded.append(float(value))
    return encoded


def decode_action(action_values):
    action = {}
    for index, key in enumerate(ACTION_KEYS):
        scaled_value = float(action_values[index]) * ACTION_SCALES[key]
        if key == "delta_Z":
            action[key] = int(round(scaled_value))
        else:
            action[key] = scaled_value
    return action


def validate_action(action_candidate):
    if isinstance(action_candidate, Action):
        return action_candidate
    try:
        return Action(**action_candidate)
    except Exception:
        return safe_default_action()


def load_model(model_path=None, device=None, use_heuristic=False):
    if use_heuristic:
        return HeuristicPolicy()

    if torch is None:
        raise RuntimeError("PyTorch is not installed in this environment. Use --heuristic or install torch.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        raise ValueError("A checkpoint path is required unless --heuristic is used.")

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        return TorchPolicy(model, device)

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format. Expected nn.Module or checkpoint dict.")

    state_dim = checkpoint.get("state_dim", len(STATE_KEYS))
    action_dim = checkpoint.get("action_dim", len(ACTION_KEYS))
    hidden_dims = checkpoint.get("hidden_dims", [128, 128])
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")

    if state_dict is None:
        raise KeyError("Checkpoint dict must contain model_state_dict or state_dict.")

    model = DeterministicActor(state_dim, hidden_dims, action_dim).to(device)
    model.load_state_dict(state_dict)
    return TorchPolicy(model, device)


def load_openai_policy(task_name, model_name):
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY is not set.")

    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = model_name if model_name else os.getenv("MODEL_NAME", "gpt-4.1-mini")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("The openai package is not installed in this environment.") from exc

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return OpenAIPolicy(client, model, task_name)


def log_start(task, benchmark, model):
    print(f"[START] task={task} env={benchmark} model={model}", flush=True)


def log_end(success, steps, score, rewards):
    success_str = str(bool(success)).lower()
    rewards_str = ",".join(f"{float(reward):.2f}" for reward in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def log_step(trajectory, state, action, reward, next_state, info, step_num=None, done=False, error=None):
    step_num = len(trajectory) + 1 if step_num is None else step_num

    # Serialize action as compact JSON
    action_dict = action.model_dump() if isinstance(action, Action) else dict(action)
    action_str = json.dumps(action_dict, separators=(",", ":"))

    # Extract key metrics for trajectory capture
    next_state_dict = next_state.model_dump() if isinstance(next_state, Observation) else dict(next_state)

    done_str = str(bool(done)).lower()
    error_value = error if error else "null"
    print(
        f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={error_value}",
        flush=True,
    )

    trajectory.append(
        {
            "state": state.model_dump() if isinstance(state, Observation) else dict(state),
            "action": action_dict,
            "reward": float(reward),
            "next_state": next_state_dict,
            "info": info.model_dump() if hasattr(info, "model_dump") else info,
            "error": error,
        }
    )


def run_episode(env, agent, max_steps=None):
    state = env.reset()
    done = False
    total_reward = 0.0
    trajectory = []
    agent.reset()
    step_num = 0

    while not done:
        step_num += 1
        action = agent.act(state, trajectory)
        next_state, reward, done, info = env.step(action)
        log_step(trajectory, state, action, reward, next_state, info, step_num=step_num, done=done)

        state = next_state
        total_reward += reward

        if max_steps is not None and env.step_count >= max_steps:
            done = True

    final_physics = env.physics
    final_constraints = env.constraints

    return {
        "total_reward": total_reward,
        "trajectory": trajectory,
        "final_state": state.model_dump() if isinstance(state, Observation) else dict(state),
        "final_physics": dict(final_physics),
        "final_constraints": dict(final_constraints),
        "feasible_score": grade_feasibility(final_physics, final_constraints),
        "pr_score": grade_target_pr(final_physics, final_constraints),
        "efficiency_score": grade_efficiency(final_physics, final_constraints),
    }


def evaluate_agent(agent, task_name, num_episodes=10, max_steps=None):
    episode_results = []

    for _ in range(num_episodes):
        env = BladeLabEnv(task_name=task_name)
        result = run_episode(env, agent, max_steps=max_steps)
        episode_results.append(result)

    rewards = [result["total_reward"] for result in episode_results]
    prs = [result["final_physics"]["pressure_ratio"] for result in episode_results]
    efficiencies = [result["final_physics"]["efficiency"] for result in episode_results]
    mass_flows = [result["final_physics"]["mass_flow"] for result in episode_results]

    def variance(values):
        return statistics.pvariance(values) if len(values) > 1 else 0.0

    summary = {
        "reward_mean": statistics.mean(rewards),
        "reward_variance": variance(rewards),
        "pr_mean": statistics.mean(prs),
        "pr_variance": variance(prs),
        "efficiency_mean": statistics.mean(efficiencies),
        "efficiency_variance": variance(efficiencies),
        "mass_flow_mean": statistics.mean(mass_flows),
        "mass_flow_variance": variance(mass_flows),
        "episodes": episode_results,
    }
    return summary


def plot_trajectory(trajectory, title="Rollout Trajectory"):
    import matplotlib.pyplot as plt

    steps = list(range(len(trajectory)))
    prs = [step["next_state"]["pressure_ratio"] for step in trajectory]
    efficiencies = [step["next_state"]["efficiency"] for step in trajectory]
    mass_flows = [step["next_state"]["mass_flow"] for step in trajectory]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, prs, label="Pressure Ratio")
    plt.plot(steps, efficiencies, label="Efficiency")
    plt.plot(steps, mass_flows, label="Mass Flow")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_episode_result(result):
    physics = result["final_physics"]
    constraints = result["final_constraints"]

    print("--- FINAL RESULTS ---")
    print(f"Total Reward: {result['total_reward']:.4f}")
    print(f"Pressure Ratio: {physics['pressure_ratio']:.6f}")
    print(f"Efficiency: {physics['efficiency']:.6f}")
    print(f"Mass Flow: {physics['mass_flow']:.6f}")
    print(f"Feasible: {constraints['feasible']}")
    print(f"Choke Margin: {constraints['choke_margin']:.6f}")
    print(f"Surge Margin: {constraints['surge_margin']:.6f}")
    print(f"Feasibility Score: {result['feasible_score']:.3f}")
    print(f"PR Score: {result['pr_score']:.3f}")
    print(f"Efficiency Score: {result['efficiency_score']:.3f}")
    print(f"Trajectory Length: {len(result['trajectory'])}")


def print_evaluation_summary(summary):
    print("--- EVALUATION SUMMARY ---")
    print(f"Reward Mean: {summary['reward_mean']:.4f}")
    print(f"Reward Variance: {summary['reward_variance']:.4f}")
    print(f"PR Mean: {summary['pr_mean']:.6f}")
    print(f"PR Variance: {summary['pr_variance']:.6f}")
    print(f"Efficiency Mean: {summary['efficiency_mean']:.6f}")
    print(f"Efficiency Variance: {summary['efficiency_variance']:.6f}")
    print(f"Mass Flow Mean: {summary['mass_flow_mean']:.6f}")
    print(f"Mass Flow Variance: {summary['mass_flow_variance']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic OpenEnv-style inference rollout.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the trained policy checkpoint.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model for API baseline runs.")
    parser.add_argument("--task", type=str, default="target_pr_efficiency", help="Environment task name.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per episode.")
    parser.add_argument("--plot", action="store_true", help="Plot the first episode trajectory.")
    parser.add_argument("--heuristic", action="store_true", help="Use deterministic heuristic fallback instead of a checkpoint.")
    parser.add_argument("--openai", action="store_true", help="Use the OpenAI API policy baseline.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.openai and args.checkpoint is None:
        args.heuristic = True

    model_label = args.model if args.openai else (args.checkpoint or "heuristic")
    log_start(args.task, "turbodesigner2", model_label)

    success = False
    steps = 0
    score = 0.0
    rewards = []

    try:
        if args.openai:
            agent = Agent(load_openai_policy(args.task, args.model))
        else:
            agent = Agent(load_model(args.checkpoint, use_heuristic=args.heuristic))

        summary = evaluate_agent(
            agent=agent,
            task_name=args.task,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
        )

        if not summary.get("episodes"):
            return

        first_episode = summary["episodes"][0]
        task = get_task(args.task)
        successes = [
            task.is_success(result["final_physics"], result["final_constraints"])
            for result in summary["episodes"]
        ]

        if args.task == "target_pr":
            score = statistics.mean(result["pr_score"] for result in summary["episodes"])
        elif args.task == "target_pr_efficiency":
            score = statistics.mean(result["efficiency_score"] for result in summary["episodes"])
        else:
            score = statistics.mean(result["feasible_score"] for result in summary["episodes"])

        success = all(successes)
        steps = sum(len(result["trajectory"]) for result in summary["episodes"])
        rewards = [
            float(step["reward"])
            for result in summary["episodes"]
            for step in result["trajectory"]
        ]

        if args.plot:
            plot_trajectory(first_episode["trajectory"], title=f"Rollout Trajectory - {args.task}")
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
