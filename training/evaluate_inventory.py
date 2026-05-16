"""Evaluate inventory policies against common baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import pandas as pd

from agents.fixed_order_agent import FixedOrderAgent
from agents.greedy_agent import GreedyAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from analytics.metrics import summarize_episode, summarize_policy
from analytics.plots import plot_policy_comparison
from training.train_inventory import build_env, load_config


class Policy(Protocol):
    """Protocol implemented by all inventory policies."""

    def choose_action(self, state: tuple[int, ...], explore: bool = False) -> int:
        """Choose an action for the supplied state."""


def run_episode(env, policy: Policy, seed: int | None = None) -> pd.DataFrame:
    """Run one evaluation episode and return a step trace."""

    state, _ = env.reset(seed=seed)
    done = False
    rows = []

    while not done:
        action = policy.choose_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        rows.append({"state": state, "next_state": next_state, **info})
        state = next_state

    return pd.DataFrame(rows)


def evaluate_policy(name: str, policy: Policy, config: dict, episodes: int) -> dict:
    """Evaluate a policy over deterministic seed offsets."""

    episode_metrics = []
    base_seed = int(config["environment"].get("seed") or 0)

    for index in range(episodes):
        env = build_env(config)
        trace = run_episode(env, policy, seed=base_seed + index)
        episode_metrics.append(summarize_episode(trace))

    return summarize_policy(name, episode_metrics).__dict__


def evaluate(config: dict, checkpoint_path: str | Path | None = None) -> pd.DataFrame:
    """Evaluate Q-learning, random, and greedy policies."""

    env = build_env(config)
    episodes = int(config["training"]["evaluation_episodes"])
    checkpoint = checkpoint_path or config["training"]["checkpoint_path"]

    learned_agent = QLearningAgent.load(checkpoint)
    learned_policy_name = config["agent"].get("algorithm", "q_learning")
    random_agent = RandomAgent(env.action_space, seed=101)
    fixed_agent = FixedOrderAgent(env.action_space, order_quantity=20)
    greedy_agent = GreedyAgent(
        env.action_space,
        inventory_bucket_size=env.config.inventory_bucket_size,
    )

    policies: list[tuple[str, Policy]] = [
        (learned_policy_name, learned_agent),
        ("random", random_agent),
        ("fixed_order_20", fixed_agent),
        ("greedy_reorder", greedy_agent),
    ]
    return pd.DataFrame(
        [evaluate_policy(name, policy, config, episodes) for name, policy in policies]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inventory_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="artifacts/evaluation_summary.csv")
    parser.add_argument("--plot", default="artifacts/policy_comparison.png")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = evaluate(config, args.checkpoint)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    plot_policy_comparison(summary, args.plot)
    print(summary.to_string(index=False))
    print(f"Saved evaluation summary to {args.output}")
    print(f"Saved comparison plot to {args.plot}")


if __name__ == "__main__":
    main()
