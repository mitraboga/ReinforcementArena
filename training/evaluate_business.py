"""Evaluate trained tabular policies for business environments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.cashflow_baseline_agent import CashFlowRuleAgent
from agents.pricing_baseline_agent import PricingRuleAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from training.train_inventory import load_config
from training.train_tabular import build_business_env


def run_episode(env, policy, seed: int | None = None) -> pd.DataFrame:
    state, _ = env.reset(seed=seed)
    done = False
    rows = []
    while not done:
        action = policy.choose_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        rows.append({"state": state, "next_state": next_state, **info})
        state = next_state
    return pd.DataFrame(rows)


def summarize_trace(env_type: str, trace: pd.DataFrame) -> dict[str, float]:
    summary = {"total_reward": float(trace["reward"].sum())}
    if env_type == "cashflow":
        summary.update(
            {
                "final_cash": float(trace["cash"].iloc[-1]),
                "final_debt": float(trace["debt"].iloc[-1]),
                "final_emergency_fund": float(trace["emergency_fund"].iloc[-1]),
                "final_net_worth": float(trace["net_worth"].iloc[-1]),
                "liquidity_penalty": float(trace["liquidity_penalty"].sum()),
            }
        )
    elif env_type == "pricing":
        total_demand = trace["demand"].sum()
        unmet_demand = trace["unmet_demand"].sum()
        summary.update(
            {
                "revenue": float(trace["revenue"].sum()),
                "units_sold": float(trace["units_sold"].sum()),
                "average_price": float(trace["price"].mean()),
                "service_level": float(1.0 if total_demand == 0 else 1 - unmet_demand / total_demand),
                "ending_inventory": float(trace["inventory"].iloc[-1]),
            }
        )
    return summary


def summarize_policy(env_type: str, name: str, episode_metrics: list[dict[str, float]]) -> dict:
    metrics = pd.DataFrame(episode_metrics)
    row = {"policy": name, "episodes": len(metrics)}
    for column in metrics.columns:
        row[f"average_{column}"] = float(metrics[column].mean())
    return row


def baseline_for(env_type: str, env, config: dict):
    if env_type == "cashflow":
        return CashFlowRuleAgent(
            cash_bucket_size=config["environment"]["cash_bucket_size"],
            risk_cash_threshold=config["environment"]["risk_cash_threshold"],
        )
    if env_type == "pricing":
        return PricingRuleAgent()
    raise ValueError(f"No business baseline for {env_type}")


def evaluate(config: dict, checkpoint_path: str | Path | None = None) -> pd.DataFrame:
    env_type = config.get("environment_type", "inventory")
    env = build_business_env(config)
    checkpoint = checkpoint_path or config["training"]["checkpoint_path"]
    learned_agent = QLearningAgent.load(checkpoint)
    random_agent = RandomAgent(env.action_space, seed=404)
    rule_agent = baseline_for(env_type, env, config)
    episodes = int(config["training"]["evaluation_episodes"])
    base_seed = int(config["environment"].get("seed") or 0)
    policies = [
        (config["agent"].get("algorithm", "q_learning"), learned_agent),
        ("random", random_agent),
        ("business_rule", rule_agent),
    ]
    rows = []

    for name, policy in policies:
        metrics = []
        for index in range(episodes):
            episode_env = build_business_env(config)
            trace = run_episode(episode_env, policy, seed=base_seed + index)
            metrics.append(summarize_trace(env_type, trace))
        rows.append(summarize_policy(env_type, name, metrics))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    env_type = config.get("environment_type", "business")
    output = args.output or f"artifacts/{env_type}_evaluation_summary.csv"
    summary = evaluate(config, args.checkpoint)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved evaluation summary to {output}")


if __name__ == "__main__":
    main()

