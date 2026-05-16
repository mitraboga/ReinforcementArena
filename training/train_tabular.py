"""Generic tabular training for business environments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from analytics.plots import plot_training_rewards
from environments.cashflow_env import CashFlowConfig, CashFlowEnv
from environments.inventory_env import InventoryConfig, InventoryEnv
from environments.pricing_env import PricingConfig, PricingEnv
from training.train_inventory import load_config


def build_business_env(config: dict[str, Any]):
    """Build an environment from a config with ``environment_type``."""

    env_type = config.get("environment_type", "inventory")
    env_config = dict(config["environment"])
    if env_type == "inventory":
        env_config["reorder_quantities"] = tuple(env_config["reorder_quantities"])
        return InventoryEnv(InventoryConfig(**env_config))
    if env_type == "cashflow":
        env_config["actions"] = tuple(env_config["actions"])
        return CashFlowEnv(CashFlowConfig(**env_config))
    if env_type == "pricing":
        env_config["price_actions"] = tuple(env_config["price_actions"])
        return PricingEnv(PricingConfig(**env_config))
    raise ValueError(f"Unsupported environment_type: {env_type}")


def build_tabular_agent(config: dict[str, Any], env) -> QLearningAgent | SARSAAgent:
    algorithm = config["agent"].get("algorithm", "q_learning")
    agent_config = {key: value for key, value in config["agent"].items() if key != "algorithm"}
    if algorithm == "q_learning":
        return QLearningAgent(actions=env.action_space, **agent_config)
    if algorithm == "sarsa":
        return SARSAAgent(actions=env.action_space, **agent_config)
    raise ValueError(f"Unsupported tabular algorithm: {algorithm}")


def train(config: dict[str, Any]) -> tuple[QLearningAgent | SARSAAgent, pd.DataFrame]:
    env = build_business_env(config)
    agent = build_tabular_agent(config, env)
    episodes = int(config["training"]["episodes"])
    rows = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        action = agent.choose_action(state, explore=True)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = None if done else agent.choose_action(next_state, explore=True)
            if isinstance(agent, SARSAAgent):
                agent.update_sarsa(state, action, reward, next_state, next_action, done)
            else:
                agent.update(state, action, reward, next_state, done)
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rows.append(
            {
                "episode": episode,
                "steps": steps,
                "total_reward": total_reward,
                "epsilon": agent.epsilon,
                "q_entries": len(agent.q),
            }
        )
        if episode == 1 or episode % 500 == 0:
            print(f"Episode {episode}/{episodes} reward={total_reward:.2f}")

    return agent, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--history", default=None)
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    env_type = config.get("environment_type", "inventory")
    agent, history = train(config)
    checkpoint_path = config["training"]["checkpoint_path"]
    history_path = args.history or f"artifacts/{env_type}_training_history.csv"
    plot_path = args.plot or f"artifacts/{env_type}_training_rewards.png"
    agent.save(checkpoint_path)
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    plot_training_rewards(history, plot_path)
    print(f"Saved agent to {checkpoint_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
