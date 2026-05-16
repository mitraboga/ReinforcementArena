"""Train a DQN agent on the inventory environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.dqn_agent import DQNAgent
from analytics.plots import plot_training_rewards
from training.train_inventory import load_config
from training.train_tabular import build_business_env


def train(config: dict) -> tuple[DQNAgent, pd.DataFrame]:
    env = build_business_env(config)
    state, _ = env.reset()
    agent_config = {
        key: value for key, value in config["agent"].items() if key != "algorithm"
    }
    agent = DQNAgent(state_size=len(state), actions=env.action_space, **agent_config)
    rows = []

    for episode in range(1, int(config["training"]["episodes"]) + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        losses = []
        steps = 0

        while not done:
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rows.append(
            {
                "episode": episode,
                "steps": steps,
                "total_reward": total_reward,
                "epsilon": agent.epsilon,
                "loss": sum(losses) / len(losses) if losses else None,
                "replay_size": len(agent.memory),
            }
        )
        if episode == 1 or episode % 50 == 0:
            print(f"Episode {episode}/{config['training']['episodes']} reward={total_reward:.2f}")

    return agent, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inventory_dqn_config.yaml")
    parser.add_argument("--history", default="artifacts/dqn_inventory_training_history.csv")
    parser.add_argument("--plot", default="artifacts/dqn_inventory_training_rewards.png")
    args = parser.parse_args()

    config = load_config(args.config)
    agent, history = train(config)
    checkpoint_path = config["training"]["checkpoint_path"]
    agent.save(checkpoint_path)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history, index=False)
    plot_training_rewards(history, args.plot)
    print(f"Saved DQN checkpoint to {checkpoint_path}")
    print(f"Saved training history to {args.history}")


if __name__ == "__main__":
    main()
