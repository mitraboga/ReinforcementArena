"""Run inventory experiments for all configured learning algorithms."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analytics.plots import plot_policy_comparison, plot_training_rewards
from training.evaluate_inventory import evaluate
from training.train_inventory import load_config, train


def run_experiment(config_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Train and evaluate one inventory experiment."""

    config = load_config(config_path)
    algorithm = config["agent"].get("algorithm", "q_learning")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent, history = train(config)
    checkpoint_path = config["training"]["checkpoint_path"]
    agent.save(checkpoint_path)

    history_path = output_dir / f"{algorithm}_training_history.csv"
    plot_path = output_dir / f"{algorithm}_training_rewards.png"
    history.to_csv(history_path, index=False)
    plot_training_rewards(history, plot_path)

    summary = evaluate(config, checkpoint_path)
    summary["experiment"] = algorithm
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/inventory_config.yaml", "configs/inventory_sarsa_config.yaml"],
    )
    parser.add_argument("--output-dir", default="artifacts/experiments")
    args = parser.parse_args()

    summaries = [run_experiment(config_path, args.output_dir) for config_path in args.configs]
    combined = pd.concat(summaries, ignore_index=True)
    output_dir = Path(args.output_dir)
    combined_path = output_dir / "inventory_experiment_summary.csv"
    plot_path = output_dir / "inventory_experiment_comparison.png"
    combined.to_csv(combined_path, index=False)
    plot_policy_comparison(combined, plot_path)
    print(combined.to_string(index=False))
    print(f"Saved combined summary to {combined_path}")
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
