"""Rule-based baseline policies for cash allocation."""

from __future__ import annotations


class CashFlowRuleAgent:
    """Prioritizes liquidity, then expensive debt, then growth investment."""

    def __init__(self, cash_bucket_size: int = 1000, risk_cash_threshold: float = 2500.0):
        self.cash_bucket_size = cash_bucket_size
        self.risk_cash_threshold = risk_cash_threshold

    def choose_action(self, state: tuple[int, ...], explore: bool = False) -> str:
        cash = state[0] * self.cash_bucket_size
        debt_bucket = state[1]
        if cash < self.risk_cash_threshold:
            return "hold_cash"
        if debt_bucket > 5:
            return "pay_debt"
        return "invest_marketing"

