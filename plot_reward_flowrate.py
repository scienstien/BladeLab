import matplotlib.pyplot as plt
from env.core_env import BladeLabEnv
from env.constraints import check_constraints
from env.reward import compute_reward
import random


def random_action():
    return {
        "delta_radius": random.uniform(-0.002, 0.002),
        "delta_angle": random.uniform(-1.0, 1.0),
        "delta_thickness": random.uniform(-0.0002, 0.0002),
    }


def run_episode(env, steps=30):
    """Run an episode and collect reward and flowrate data."""
    obs = env.reset()

    flowrates = []
    rewards = []
    efficiencies = []
    pressure_ratios = []

    prev_physics = env.prev_physics.copy()

    for step in range(steps):
        action = random_action()
        obs, reward, done, _ = env.step(action)

        physics = env.prev_physics
        flowrates.append(physics["mass_flow"])
        rewards.append(reward)
        efficiencies.append(physics["efficiency"])
        pressure_ratios.append(physics["pressure_ratio"])

        prev_physics = physics.copy()

        if done:
            break

    return flowrates, rewards, efficiencies, pressure_ratios


def plot_reward_vs_flowrate(flowrates, rewards):
    """Plot reward vs mass flow rate."""
    plt.figure(figsize=(10, 6))
    plt.scatter(flowrates, rewards, c=range(len(rewards)), cmap='viridis', s=100, alpha=0.6)
    plt.plot(flowrates, rewards, alpha=0.3, linewidth=1)

    plt.xlabel("Mass Flow Rate (kg/s)", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Reward vs Mass Flow Rate", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Step')
    plt.tight_layout()
    plt.savefig("reward_vs_flowrate.png", dpi=150)
    print("Saved: reward_vs_flowrate.png")
    plt.show()


def plot_reward_over_time(rewards):
    """Plot reward progression over steps."""
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(rewards)), rewards, marker='o', markersize=4)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Reward Progression Over Time", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_over_time.png", dpi=150)
    print("Saved: reward_over_time.png")
    plt.show()


def plot_flowrate_distribution(flowrates):
    """Plot distribution of flow rates."""
    plt.figure(figsize=(10, 4))
    plt.hist(flowrates, bins=15, edgecolor='black', alpha=0.7)
    plt.xlabel("Mass Flow Rate (kg/s)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Mass Flow Rate Distribution", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("flowrate_distribution.png", dpi=150)
    print("Saved: flowrate_distribution.png")
    plt.show()


def main():
    env = BladeLabEnv()

    print("Running episode with random policy...")
    flowrates, rewards, efficiencies, pressure_ratios = run_episode(env, steps=30)

    print(f"\n--- Episode Statistics ---")
    print(f"Mean Reward: {sum(rewards)/len(rewards):.4f}")
    print(f"Min Reward: {min(rewards):.4f}")
    print(f"Max Reward: {max(rewards):.4f}")
    print(f"Mean Flow Rate: {sum(flowrates)/len(flowrates):.4f} kg/s")
    print(f"Mean Efficiency: {sum(efficiencies)/len(efficiencies):.4f}")
    print(f"Mean Pressure Ratio: {sum(pressure_ratios)/len(pressure_ratios):.4f}")

    # Generate plots
    plot_reward_vs_flowrate(flowrates, rewards)
    plot_reward_over_time(rewards)
    plot_flowrate_distribution(flowrates)

    print("\nDone!")


if __name__ == "__main__":
    main()
