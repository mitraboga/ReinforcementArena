"""Dynamic pricing environment for revenue and margin optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PricingConfig:
    """Configuration for price optimization decisions."""

    initial_inventory: int = 160
    unit_cost: float = 10.0
    base_price: float = 25.0
    reference_price: float = 25.0
    base_demand: float = 26.0
    demand_std: float = 4.0
    price_elasticity: float = 1.4
    holding_cost: float = 0.35
    stockout_penalty: float = 6.0
    episode_length: int = 45
    price_actions: tuple[int, ...] = (18, 22, 25, 28, 32, 36)
    inventory_bucket_size: int = 10
    demand_bucket_size: int = 5
    price_bucket_size: int = 5
    seed: int | None = None


class PricingEnv:
    """Environment where an agent sets prices for a finite inventory product."""

    def __init__(self, config: PricingConfig | None = None):
        self.config = config or PricingConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.action_space = tuple(self.config.price_actions)
        self.inventory = self.config.initial_inventory
        self.day = 0
        self.last_demand = int(round(self.config.base_demand))
        self.last_price = int(round(self.config.base_price))

    def reset(self, seed: int | None = None) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.inventory = self.config.initial_inventory
        self.day = 0
        self.last_demand = int(round(self.config.base_demand))
        self.last_price = int(round(self.config.base_price))
        return self.get_state(), self._info()

    def step(self, action: int) -> tuple[tuple[int, int, int, int], float, bool, dict[str, Any]]:
        if action not in self.action_space:
            raise ValueError(f"Invalid price {action}. Allowed: {self.action_space}")

        expected_demand = self._expected_demand(action)
        demand = max(0, int(round(self.rng.normal(expected_demand, self.config.demand_std))))
        units_sold = min(self.inventory, demand)
        unmet_demand = max(demand - units_sold, 0)
        revenue = units_sold * action
        variable_cost = units_sold * self.config.unit_cost
        holding_cost = max(self.inventory - units_sold, 0) * self.config.holding_cost
        stockout_cost = unmet_demand * self.config.stockout_penalty
        reward = revenue - variable_cost - holding_cost - stockout_cost

        self.inventory -= units_sold
        self.day += 1
        self.last_demand = demand
        self.last_price = action
        done = self.day >= self.config.episode_length or self.inventory <= 0

        info = self._info()
        info.update(
            {
                "action": action,
                "price": action,
                "expected_demand": expected_demand,
                "demand": demand,
                "units_sold": units_sold,
                "unmet_demand": unmet_demand,
                "revenue": revenue,
                "variable_cost": variable_cost,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "reward": reward,
            }
        )
        return self.get_state(), float(reward), done, info

    def get_state(self) -> tuple[int, int, int, int]:
        return (
            self.inventory // self.config.inventory_bucket_size,
            self.last_demand // self.config.demand_bucket_size,
            self.last_price // self.config.price_bucket_size,
            self.day,
        )

    def _expected_demand(self, price: int) -> float:
        price_ratio = max(price / self.config.reference_price, 0.01)
        return self.config.base_demand * price_ratio ** (-self.config.price_elasticity)

    def _info(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "inventory": self.inventory,
            "last_demand": self.last_demand,
            "last_price": self.last_price,
        }
