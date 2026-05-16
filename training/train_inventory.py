"""Train a Q-learning agent on the inventory environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from analytics.plots import plot_training_rewards
from environments.inventory_env import InventoryConfig, InventoryEnv


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_env(config: dict[str, Any]) -> InventoryEnv:
    """Create the inventory environment from config data."""

    env_config = dict(config["environment"])
    env_config["reorder_quantities"] = tuple(env_config["reorder_quantities"])
    return InventoryEnv(InventoryConfig(**env_config))


def build_agent(config: dict[str, Any], env: InventoryEnv) -> QLearningAgent | SARSAAgent:
    """Create a temporal-difference learning agent from config data."""

    algorithm = config["agent"].get("algorithm", "q_learning")
    agent_config = {
        key: value for key, value in config["agent"].items() if key != "algorithm"
    }
    if algorithm == "q_learning":
        return QLearningAgent(actions=env.action_space, **agent_config)
    if algorithm == "sarsa":
        return SARSAAgent(actions=env.action_space, **agent_config)
    raise ValueError(f"Unsupported agent algorithm: {algorithm}")


def train(config: dict[str, Any]) -> tuple[QLearningAgent | SARSAAgent, pd.DataFrame]:
    """Train a temporal-difference policy and return the agent and history."""

    env = build_env(config)
    agent = build_agent(config, env)
    episodes = int(config["training"]["episodes"])
    rows = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        action = agent.choose_action(state, explore=True)

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
                "q_states": len({key[0] for key in agent.q}),
                "q_entries": len(agent.q),
            }
        )

        if episode == 1 or episode % 250 == 0:
            print(f"Episode {episode}/{episodes} reward={total_reward:.2f}")

    history = pd.DataFrame(rows)
    return agent, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inventory_config.yaml")
    parser.add_argument("--history", default="artifacts/training_history.csv")
    parser.add_argument("--plot", default="artifacts/training_rewards.png")
    args = parser.parse_args()

    config = load_config(args.config)
    agent, history = train(config)

    checkpoint_path = config["training"]["checkpoint_path"]
    agent.save(checkpoint_path)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history, index=False)
    plot_training_rewards(history, args.plot)

    print(f"Saved agent to {checkpoint_path}")
    print(f"Saved training history to {args.history}")
    print(f"Saved training plot to {args.plot}")


if __name__ == "__main__":
    main()
