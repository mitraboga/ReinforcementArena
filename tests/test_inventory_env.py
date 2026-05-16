"""Unit tests for the inventory environment and agents."""

from __future__ import annotations

import unittest

from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from environments.inventory_env import InventoryConfig, InventoryEnv


class InventoryEnvTests(unittest.TestCase):
    def test_reset_returns_discrete_state(self) -> None:
        env = InventoryEnv(InventoryConfig(seed=1))

        state, info = env.reset(seed=1)

        self.assertEqual(len(state), 4)
        self.assertEqual(info["inventory"], 40)
        self.assertEqual(info["cash"], 10000.0)

    def test_step_rejects_invalid_action(self) -> None:
        env = InventoryEnv(InventoryConfig(seed=1))
        env.reset(seed=1)

        with self.assertRaises(ValueError):
            env.step(999)

    def test_step_respects_pipeline_limit(self) -> None:
        env = InventoryEnv(
            InventoryConfig(
                max_pending_orders=5,
                reorder_quantities=(0, 10),
                seed=1,
            )
        )
        env.reset(seed=1)

        _, _, _, info = env.step(10)

        self.assertEqual(info["feasible_order"], 5)

    def test_pipeline_orders_arrive_after_lead_time(self) -> None:
        env = InventoryEnv(
            InventoryConfig(
                initial_inventory=0,
                mean_demand=0,
                demand_std=0,
                supplier_lead_time=1,
                reorder_quantities=(0, 10),
                seed=1,
            )
        )
        env.reset(seed=1)

        env.step(10)
        _, _, _, info = env.step(0)

        self.assertEqual(info["received_units"], 10)


class QLearningAgentTests(unittest.TestCase):
    def test_unknown_q_value_defaults_to_zero(self) -> None:
        agent = QLearningAgent(actions=(0, 10), seed=1)

        self.assertEqual(agent.get_q_value((1, 2, 3), 10), 0.0)

    def test_update_changes_q_value(self) -> None:
        agent = QLearningAgent(actions=(0, 10), alpha=0.5, gamma=0.9, seed=1)

        agent.update((1, 2, 3), 10, reward=100.0, next_state=(1, 2, 4), done=True)

        self.assertEqual(agent.get_q_value((1, 2, 3), 10), 50.0)

    def test_sarsa_update_uses_next_policy_action(self) -> None:
        agent = SARSAAgent(actions=(0, 10), alpha=0.5, gamma=0.9, seed=1)
        agent.q[((1, 2, 4), 0)] = 20.0

        agent.update_sarsa(
            (1, 2, 3),
            10,
            reward=100.0,
            next_state=(1, 2, 4),
            next_action=0,
            done=False,
        )

        self.assertEqual(agent.get_q_value((1, 2, 3), 10), 59.0)


if __name__ == "__main__":
    unittest.main()
