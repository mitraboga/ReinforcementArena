"""Cash allocation environment for finance optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CashFlowConfig:
    """Configuration for monthly cash allocation decisions."""

    initial_cash: float = 5000.0
    initial_debt: float = 12000.0
    emergency_fund: float = 2000.0
    monthly_revenue: float = 8000.0
    revenue_std: float = 900.0
    operating_expenses: float = 4500.0
    expense_std: float = 400.0
    monthly_interest_rate: float = 0.015
    marketing_roi: float = 1.35
    savings_return: float = 0.003
    risk_cash_threshold: float = 2500.0
    risk_penalty: float = 750.0
    allocation_amount: float = 1000.0
    episode_length: int = 24
    actions: tuple[str, ...] = ("pay_debt", "invest_marketing", "save_cash", "hold_cash")
    cash_bucket_size: int = 1000
    debt_bucket_size: int = 1000
    fund_bucket_size: int = 500
    seed: int | None = None


class CashFlowEnv:
    """Environment where an agent allocates surplus cash across finance choices."""

    def __init__(self, config: CashFlowConfig | None = None):
        self.config = config or CashFlowConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.action_space = tuple(self.config.actions)
        self.cash = self.config.initial_cash
        self.debt = self.config.initial_debt
        self.emergency_fund = self.config.emergency_fund
        self.month = 0

    def reset(self, seed: int | None = None) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.cash = self.config.initial_cash
        self.debt = self.config.initial_debt
        self.emergency_fund = self.config.emergency_fund
        self.month = 0
        return self.get_state(), self._info()

    def step(self, action: str) -> tuple[tuple[int, int, int, int], float, bool, dict[str, Any]]:
        if action not in self.action_space:
            raise ValueError(f"Invalid cash allocation action {action}. Allowed: {self.action_space}")

        starting_net_worth = self.net_worth
        revenue = max(0.0, float(self.rng.normal(self.config.monthly_revenue, self.config.revenue_std)))
        expenses = max(0.0, float(self.rng.normal(self.config.operating_expenses, self.config.expense_std)))
        self.cash += revenue - expenses

        interest = self.debt * self.config.monthly_interest_rate
        self.debt += interest
        allocation = min(self.config.allocation_amount, max(self.cash, 0.0))
        marketing_gain = 0.0
        savings_gain = 0.0
        debt_payment = 0.0
        saved_cash = 0.0

        if action == "pay_debt":
            debt_payment = min(allocation, self.debt)
            self.cash -= debt_payment
            self.debt -= debt_payment
        elif action == "invest_marketing":
            self.cash -= allocation
            marketing_gain = allocation * self.config.marketing_roi
            self.cash += marketing_gain
        elif action == "save_cash":
            self.cash -= allocation
            saved_cash = allocation
            savings_gain = (self.emergency_fund + saved_cash) * self.config.savings_return
            self.emergency_fund += saved_cash + savings_gain

        liquidity_penalty = self.config.risk_penalty if self.cash < self.config.risk_cash_threshold else 0.0
        reward = (self.net_worth - starting_net_worth) - liquidity_penalty
        self.month += 1
        done = self.month >= self.config.episode_length or self.cash < -self.config.operating_expenses

        info = self._info()
        info.update(
            {
                "action": action,
                "revenue": revenue,
                "expenses": expenses,
                "interest": interest,
                "allocation": allocation,
                "debt_payment": debt_payment,
                "marketing_gain": marketing_gain,
                "saved_cash": saved_cash,
                "savings_gain": savings_gain,
                "liquidity_penalty": liquidity_penalty,
                "reward": reward,
                "net_worth": self.net_worth,
            }
        )
        return self.get_state(), float(reward), done, info

    @property
    def net_worth(self) -> float:
        return self.cash + self.emergency_fund - self.debt

    def get_state(self) -> tuple[int, int, int, int]:
        return (
            int(self.cash // self.config.cash_bucket_size),
            int(self.debt // self.config.debt_bucket_size),
            int(self.emergency_fund // self.config.fund_bucket_size),
            self.month,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "cash": self.cash,
            "debt": self.debt,
            "emergency_fund": self.emergency_fund,
        }

