import argparse
import json
import os
import statistics

from env.core_env import BladeLabEnv
from env.graders import grade_efficiency, grade_feasibility, grade_target_pr
from env.models import Action, Observation
from env.tasks import get_task


class StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


class Agent:
    def __init__(self, policy):
        self.policy = policy

    def act(self, observation, trajectory=None):
        return validate_action(self.policy(observation, trajectory))

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
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
                max_tokens=120,
            )
        except Exception as exc:
            raise RuntimeError("LLM call failed") from exc

        text = response.choices[0].message.content if response and response.choices else ""
        if not text:
            raise RuntimeError("LLM call failed")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM call failed") from exc

        return validate_action(parsed)


def validate_action(action_candidate):
    if isinstance(action_candidate, Action):
        return action_candidate
    try:
        return Action(**action_candidate)
    except Exception as exc:
        raise RuntimeError("LLM call failed") from exc


def log_start(task, benchmark, model):
    print(f"[START] task={task} env={benchmark} model={model}")


def log_end(success, steps, score, rewards):
    rewards_str = "[" + ",".join(f"{reward:.2f}" for reward in rewards) + "]" if rewards else "[]"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")


def log_step(trajectory, state, action, reward, next_state, info, step_num=None, done=False, error=None):
    step_num = len(trajectory) + 1 if step_num is None else step_num
    action_dict = action.model_dump() if isinstance(action, Action) else dict(action)
    action_str = json.dumps(action_dict, separators=(",", ":"))
    error_str = "null" if error is None else json.dumps(str(error), separators=(",", ":"))
    print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}")
    trajectory.append(
        {
            "state": state.model_dump() if isinstance(state, Observation) else dict(state),
            "action": action_dict,
            "reward": float(reward),
            "next_state": next_state.model_dump() if isinstance(next_state, Observation) else dict(next_state),
            "info": info.model_dump() if hasattr(info, "model_dump") else info,
            "error": error,
        }
    )


def run_episode(env, agent, max_steps=None):
    state = None
    total_reward = 0.0
    trajectory = []
    agent.reset()
    step_num = 0
    try:
        state = env.reset()
        done = False
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
    finally:
        env.close()


def evaluate_agent(agent, task_name, num_episodes=10, max_steps=None):
    episode_results = []
    for _ in range(num_episodes):
        env = BladeLabEnv(task_name=task_name)
        episode_results.append(run_episode(env, agent, max_steps=max_steps))

    rewards = [result["total_reward"] for result in episode_results]
    prs = [result["final_physics"]["pressure_ratio"] for result in episode_results]
    efficiencies = [result["final_physics"]["efficiency"] for result in episode_results]
    mass_flows = [result["final_physics"]["mass_flow"] for result in episode_results]

    def variance(values):
        return statistics.pvariance(values) if len(values) > 1 else 0.0

    return {
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


def parse_args():
    parser = StrictArgumentParser(description="Deterministic OpenEnv-style inference rollout.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model for policy runs.")
    parser.add_argument("--task", type=str, default="target_pr_efficiency", help="Environment task name.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per episode.")
    return parser.parse_args()


def main():
    args = None
    failure = None
    success = False
    steps = 0
    score = 0.0
    rewards = []
    task_name = "target_pr_efficiency"
    model_label = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

    try:
        args = parse_args()
        task_name = args.task
        model_label = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
    except BaseException as exc:
        failure = exc

    log_start(task_name, "turbodesigner2", model_label)

    try:
        if failure is not None:
            raise failure
        if args.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if args.max_steps is not None and args.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        from openai import OpenAI

        API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
        MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
        model = MODEL_NAME

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except Exception as exc:
            raise RuntimeError("LLM call failed") from exc

        agent = Agent(OpenAIPolicy(client, model, args.task))
        summary = evaluate_agent(
            agent=agent,
            task_name=args.task,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
        )
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
        rewards = [result["total_reward"] for result in summary["episodes"]]
    except BaseException as exc:
        failure = exc
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
