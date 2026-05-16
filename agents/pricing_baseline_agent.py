"""Rule-based baseline policies for dynamic pricing."""

from __future__ import annotations


class PricingRuleAgent:
    """Raises price when inventory is scarce and discounts when inventory is high."""

    def __init__(self, low_inventory_bucket: int = 4, high_inventory_bucket: int = 12):
        self.low_inventory_bucket = low_inventory_bucket
        self.high_inventory_bucket = high_inventory_bucket

    def choose_action(self, state: tuple[int, ...], explore: bool = False) -> int:
        inventory_bucket = state[0]
        if inventory_bucket <= self.low_inventory_bucket:
            return 36
        if inventory_bucket >= self.high_inventory_bucket:
            return 22
        return 28

