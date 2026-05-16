"""Evaluate a trained DQN inventory policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.dqn_agent import DQNAgent
from agents.fixed_order_agent import FixedOrderAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from analytics.metrics import summarize_episode, summarize_policy
from analytics.plots import plot_policy_comparison
from training.evaluate_inventory import run_episode
from training.train_inventory import load_config
from training.train_tabular import build_business_env


def evaluate(config: dict, checkpoint_path: str | Path | None = None) -> pd.DataFrame:
    env = build_business_env(config)
    checkpoint = checkpoint_path or config["training"]["checkpoint_path"]
    dqn_config = {
        key: value
        for key, value in config["agent"].items()
        if key
        not in {
            "algorithm",
            "epsilon",
            "seed",
            "replay_size",
            "batch_size",
            "target_update_frequency",
        }
    }
    learned_agent = DQNAgent.load(checkpoint, **dqn_config)
    learned_agent.epsilon = 0.0
    policies = [
        ("dqn", learned_agent),
        ("random", RandomAgent(env.action_space, seed=909)),
        ("fixed_order_20", FixedOrderAgent(env.action_space, order_quantity=20)),
        (
            "greedy_reorder",
            GreedyAgent(env.action_space, inventory_bucket_size=env.config.inventory_bucket_size),
        ),
    ]
    episodes = int(config["training"]["evaluation_episodes"])
    base_seed = int(config["environment"].get("seed") or 0)
    rows = []

    for name, policy in policies:
        episode_metrics = []
        for index in range(episodes):
            episode_env = build_business_env(config)
            trace = run_episode(episode_env, policy, seed=base_seed + index)
            episode_metrics.append(summarize_episode(trace))
        rows.append(summarize_policy(name, episode_metrics).__dict__)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inventory_dqn_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="artifacts/dqn_inventory_evaluation_summary.csv")
    parser.add_argument("--plot", default="artifacts/dqn_inventory_policy_comparison.png")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = evaluate(config, args.checkpoint)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    plot_policy_comparison(summary, args.plot)
    print(summary.to_string(index=False))
    print(f"Saved evaluation summary to {args.output}")


if __name__ == "__main__":
    main()
