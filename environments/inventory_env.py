"""Inventory management environment for tabular reinforcement learning.

The environment models a single product with stochastic demand, finite storage,
procurement cost, sales revenue, holding costs, and lost-sales penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    """Configuration for the inventory optimization environment."""

    initial_inventory: int = 40
    initial_cash: float = 10000.0
    max_inventory: int = 120
    unit_cost: float = 12.0
    selling_price: float = 25.0
    holding_cost: float = 0.5
    stockout_penalty: float = 8.0
    mean_demand: float = 22.0
    demand_std: float = 6.0
    seasonal_amplitude: float = 0.25
    seasonal_period: int = 30
    forecast_noise_std: float = 3.0
    supplier_lead_time: int = 2
    emergency_unit_cost: float = 32.0
    max_pending_orders: int = 120
    episode_length: int = 60
    reorder_quantities: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60)
    inventory_bucket_size: int = 10
    cash_bucket_size: int = 1000
    demand_bucket_size: int = 5
    pipeline_bucket_size: int = 10
    seed: int | None = None


class InventoryEnv:
    """Single-product inventory environment.

    State is returned as a discretized tuple:
        (inventory_bucket, cash_bucket, demand_forecast_bucket, pipeline_bucket)

    The raw business values are available in ``info`` from ``reset`` and ``step``.
    """

    def __init__(self, config: InventoryConfig | None = None):
        self.config = config or InventoryConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.action_space = tuple(self.config.reorder_quantities)
        self.inventory = self.config.initial_inventory
        self.cash = self.config.initial_cash
        self.day = 0
        self.demand_forecast = int(round(self.config.mean_demand))
        self.pipeline_orders: list[dict[str, int]] = []

    def reset(self, seed: int | None = None) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
        """Reset the environment and return the initial state and metadata."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.inventory = self.config.initial_inventory
        self.cash = self.config.initial_cash
        self.day = 0
        self.demand_forecast = self._sample_demand_forecast()
        self.pipeline_orders = []
        return self.get_state(), self._info()

    def step(self, action: int) -> tuple[tuple[int, int, int, int], float, bool, dict[str, Any]]:
        """Apply a reorder action and advance the environment by one period."""

        if action not in self.action_space:
            raise ValueError(f"Invalid reorder quantity {action}. Allowed: {self.action_space}")

        received_units = self._receive_pipeline_orders()
        feasible_order = min(
            action,
            self.config.max_pending_orders - self.pending_units,
            int(self.cash // self.config.unit_cost),
        )
        procurement_cost = feasible_order * self.config.unit_cost
        if feasible_order > 0:
            self.pipeline_orders.append(
                {"quantity": feasible_order, "days_remaining": self.config.supplier_lead_time}
            )
        self.cash -= procurement_cost

        demand = self._sample_demand()
        units_sold = min(self.inventory, demand)
        unmet_demand = max(demand - units_sold, 0)
        revenue = units_sold * self.config.selling_price
        emergency_cost = unmet_demand * self.config.emergency_unit_cost
        holding_cost = self.inventory_after_sales(units_sold) * self.config.holding_cost
        stockout_cost = unmet_demand * self.config.stockout_penalty

        self.inventory -= units_sold
        self.cash += revenue - holding_cost - stockout_cost - emergency_cost
        reward = revenue - procurement_cost - holding_cost - stockout_cost - emergency_cost

        self.day += 1
        self.demand_forecast = self._sample_demand_forecast()
        done = self.day >= self.config.episode_length

        info = self._info()
        info.update(
            {
                "action": action,
                "received_units": received_units,
                "feasible_order": feasible_order,
                "pending_units": self.pending_units,
                "demand": demand,
                "units_sold": units_sold,
                "unmet_demand": unmet_demand,
                "revenue": revenue,
                "procurement_cost": procurement_cost,
                "emergency_cost": emergency_cost,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "reward": reward,
            }
        )
        return self.get_state(), float(reward), done, info

    def get_state(self) -> tuple[int, int, int, int]:
        """Return the current state as a compact discrete tuple."""

        return (
            self.inventory // self.config.inventory_bucket_size,
            int(self.cash // self.config.cash_bucket_size),
            self.demand_forecast // self.config.demand_bucket_size,
            self.pending_units // self.config.pipeline_bucket_size,
        )

    @property
    def pending_units(self) -> int:
        """Return all units currently on order."""

        return sum(order["quantity"] for order in self.pipeline_orders)

    def _sample_demand(self) -> int:
        demand = self.rng.normal(self._expected_demand(), self.config.demand_std)
        return max(0, int(round(demand)))

    def _sample_demand_forecast(self) -> int:
        forecast_noise = self.rng.normal(0, self.config.forecast_noise_std)
        return max(0, int(round(self._expected_demand() + forecast_noise)))

    def _expected_demand(self) -> float:
        seasonal = 1 + self.config.seasonal_amplitude * np.sin(
            2 * np.pi * self.day / self.config.seasonal_period
        )
        return self.config.mean_demand * seasonal

    def _receive_pipeline_orders(self) -> int:
        received_units = 0
        remaining_orders = []

        for order in self.pipeline_orders:
            order["days_remaining"] -= 1
            if order["days_remaining"] <= 0:
                received_units += order["quantity"]
            else:
                remaining_orders.append(order)

        self.pipeline_orders = remaining_orders
        available_capacity = self.config.max_inventory - self.inventory
        accepted_units = min(received_units, available_capacity)
        self.inventory += accepted_units
        return accepted_units

    def inventory_after_sales(self, units_sold: int) -> int:
        return max(self.inventory - units_sold, 0)

    def _info(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "inventory": self.inventory,
            "cash": self.cash,
            "demand_forecast": self.demand_forecast,
            "pending_units": self.pending_units,
        }
