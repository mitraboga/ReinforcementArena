"""Rule-based inventory baseline policy."""

from __future__ import annotations

from typing import Iterable


class GreedyAgent:
    """Orders enough inventory to reach a target level when stock is low."""

    def __init__(
        self,
        actions: Iterable[int],
        inventory_bucket_size: int,
        reorder_point: int = 30,
        target_inventory: int = 70,
    ):
        self.actions = tuple(sorted(actions))
        self.inventory_bucket_size = inventory_bucket_size
        self.reorder_point = reorder_point
        self.target_inventory = target_inventory

    def choose_action(self, state: tuple[int, ...], explore: bool = False) -> int:
        inventory = state[0] * self.inventory_bucket_size
        if inventory >= self.reorder_point:
            return 0

        needed = max(self.target_inventory - inventory, 0)
        feasible_actions = [action for action in self.actions if action <= needed]
        if feasible_actions:
            return max(feasible_actions)
        return min(self.actions, key=lambda action: abs(action - needed))

