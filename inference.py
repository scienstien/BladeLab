import random
import matplotlib.pyplot as plt

from env.core_env import BladeLabEnv
from env.constraints import check_constraints
from env.graders import (
    grade_feasibility,
    grade_target_pr,
    grade_efficiency
)

# -------------------------------
# Simple random policy
# -------------------------------
def random_action():
    return {
        "delta_radius": random.uniform(-0.002, 0.002),
        "delta_angle": random.uniform(-1.0, 1.0),
        "delta_thickness": random.uniform(-0.0002, 0.0002),
    }


# -------------------------------
# Slightly smarter hill-climbing
# -------------------------------
def greedy_action(env, base_action, scale=1.0):
    """
    Try small variations and pick best
    """
    best_action = base_action
    best_reward = -1e9

    for _ in range(5):
        action = {
            "delta_radius": base_action["delta_radius"] + random.uniform(-scale, scale) * 0.001,
            "delta_angle": base_action["delta_angle"] + random.uniform(-scale, scale),
            "delta_thickness": base_action["delta_thickness"] + random.uniform(-scale, scale) * 0.0001,
        }

        # simulate one step (without committing)
        temp_env = BladeLabEnv()
        temp_env.state = env.state.copy()
        temp_env.prev_physics = env.prev_physics.copy()

        obs, reward, _, _ = temp_env.step(action)

        if reward > best_reward:
            best_reward = reward
            best_action = action

    return best_action


# -------------------------------
# Run one episode
# -------------------------------
def run_episode(env, steps=30, use_greedy=True):
    obs = env.reset()

    actions = []
    rewards = []

    action = random_action()

    for step in range(steps):
        if use_greedy:
            action = greedy_action(env, action)

        obs, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)

        if done:
            break

    return actions, rewards


# -------------------------------
# Plot compressor trajectory
# -------------------------------
def plot_trajectory(env):
    traj = env.get_trajectory()

    m = [p["mass_flow"] for p in traj]
    pr = [p["pressure_ratio"] for p in traj]

    plt.figure()
    plt.plot(m, pr, marker='o')
    plt.xlabel("Mass Flow")
    plt.ylabel("Pressure Ratio")
    plt.title("Compressor Map Trajectory")
    plt.grid()
    plt.show()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    env = BladeLabEnv()

    actions, rewards = run_episode(env, steps=30)

    # Final physics
    physics = env.prev_physics
    constraints = check_constraints(physics)

    # Scores
    s1 = grade_feasibility(physics, constraints)
    s2 = grade_target_pr(physics, constraints)
    s3 = grade_efficiency(physics, constraints)

    print("\n--- FINAL RESULTS ---")
    print(f"Efficiency: {physics['efficiency']:.4f}")
    print(f"Pressure Ratio: {physics['pressure_ratio']:.4f}")
    print(f"Mass Flow: {physics['mass_flow']:.4f}")

    print("\n--- SCORES ---")
    print(f"Feasibility: {s1:.3f}")
    print(f"Target PR: {s2:.3f}")
    print(f"Efficiency Task: {s3:.3f}")

    # Plot behavior
    plot_trajectory(env)