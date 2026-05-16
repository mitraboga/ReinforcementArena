"""Metrics for inventory policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated business metrics for a policy."""

    policy: str
    episodes: int
    average_reward: float
    average_final_cash: float
    average_stockout_rate: float
    average_service_level: float
    average_units_sold: float
    average_emergency_cost: float
    average_holding_cost: float
    average_ending_inventory: float


def summarize_episode(trace: pd.DataFrame) -> dict[str, float]:
    """Compute episode-level inventory metrics from a step trace."""

    total_demand = trace["demand"].sum()
    total_unmet = trace["unmet_demand"].sum()
    stockout_periods = (trace["unmet_demand"] > 0).sum()
    service_level = 1.0 if total_demand == 0 else 1.0 - (total_unmet / total_demand)

    return {
        "total_reward": float(trace["reward"].sum()),
        "final_cash": float(trace["cash"].iloc[-1]),
        "stockout_rate": float(stockout_periods / len(trace)),
        "service_level": float(service_level),
        "units_sold": float(trace["units_sold"].sum()),
        "emergency_cost": float(trace["emergency_cost"].sum()),
        "holding_cost": float(trace["holding_cost"].sum()),
        "ending_inventory": float(trace["inventory"].iloc[-1]),
    }


def summarize_policy(policy: str, episode_metrics: list[dict[str, float]]) -> EvaluationSummary:
    """Aggregate multiple episode summaries into a policy summary."""

    metrics = pd.DataFrame(episode_metrics)
    return EvaluationSummary(
        policy=policy,
        episodes=len(metrics),
        average_reward=float(metrics["total_reward"].mean()),
        average_final_cash=float(metrics["final_cash"].mean()),
        average_stockout_rate=float(metrics["stockout_rate"].mean()),
        average_service_level=float(metrics["service_level"].mean()),
        average_units_sold=float(metrics["units_sold"].mean()),
        average_emergency_cost=float(metrics["emergency_cost"].mean()),
        average_holding_cost=float(metrics["holding_cost"].mean()),
        average_ending_inventory=float(metrics["ending_inventory"].mean()),
    )
