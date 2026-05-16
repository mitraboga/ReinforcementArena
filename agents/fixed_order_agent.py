"""Fixed-order baseline policy."""

from __future__ import annotations

from typing import Iterable


class FixedOrderAgent:
    """Orders the same quantity every period."""

    def __init__(self, actions: Iterable[int], order_quantity: int = 20):
        self.actions = tuple(actions)
        if order_quantity not in self.actions:
            raise ValueError(f"{order_quantity} is not in the action space: {self.actions}")
        self.order_quantity = order_quantity

    def choose_action(self, state: tuple[int, ...], explore: bool = False) -> int:
        return self.order_quantity

