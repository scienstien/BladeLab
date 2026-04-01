import argparse
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from env.core_env import BladeLabEnv
from env.graders import grade_efficiency, grade_feasibility, grade_target_pr


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
        return decode_action(action_tensor)


class HeuristicPolicy:
    """Deterministic fallback policy for debugging when no checkpoint is available."""

    def __call__(self, state):
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


def encode_state(state):
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


def log_step(trajectory, state, action, reward, next_state, info):
    trajectory.append(
        {
            "state": dict(state),
            "action": dict(action),
            "reward": float(reward),
            "next_state": dict(next_state),
            "info": info,
        }
    )


def run_episode(env, policy, max_steps=None):
    state = env.reset()
    done = False
    total_reward = 0.0
    trajectory = []

    while not done:
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        log_step(trajectory, state, action, reward, next_state, info)

        state = next_state
        total_reward += reward

        if max_steps is not None and env.step_count >= max_steps:
            done = True

    final_physics = env.physics
    final_constraints = env.constraints

    return {
        "total_reward": total_reward,
        "trajectory": trajectory,
        "final_state": dict(state),
        "final_physics": dict(final_physics),
        "final_constraints": dict(final_constraints),
        "feasible_score": grade_feasibility(final_physics, final_constraints),
        "pr_score": grade_target_pr(final_physics, final_constraints),
        "efficiency_score": grade_efficiency(final_physics, final_constraints),
    }


def evaluate_agent(policy, task_name, num_episodes=10, max_steps=None):
    episode_results = []

    for _ in range(num_episodes):
        env = BladeLabEnv(task_name=task_name)
        result = run_episode(env, policy, max_steps=max_steps)
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
    parser.add_argument("--task", type=str, default="target_pr_efficiency", help="Environment task name.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per episode.")
    parser.add_argument("--plot", action="store_true", help="Plot the first episode trajectory.")
    parser.add_argument("--heuristic", action="store_true", help="Use deterministic heuristic fallback instead of a checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    policy = load_model(args.checkpoint, use_heuristic=args.heuristic)

    summary = evaluate_agent(
        policy=policy,
        task_name=args.task,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    first_episode = summary["episodes"][0]
    print_episode_result(first_episode)
    print()
    print_evaluation_summary(summary)

    if args.plot:
        plot_trajectory(first_episode["trajectory"], title=f"Rollout Trajectory - {args.task}")


if __name__ == "__main__":
    main()
