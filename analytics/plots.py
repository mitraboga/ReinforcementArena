"""Plotting helpers for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_rewards(history: pd.DataFrame, output_path: str | Path) -> None:
    """Save a reward curve with a rolling average."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history["episode"], history["total_reward"], alpha=0.35, label="Episode reward")
    rolling = history["total_reward"].rolling(window=100, min_periods=1).mean()
    plt.plot(history["episode"], rolling, label="100-episode rolling mean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning Inventory Training Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_policy_comparison(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Save a bar chart comparing average reward by policy."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = summary["policy"]
    if "experiment" in summary.columns:
        labels = summary["experiment"] + ":" + summary["policy"]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, summary["average_reward"])
    plt.xlabel("Policy")
    plt.ylabel("Average reward")
    plt.title("Inventory Policy Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
