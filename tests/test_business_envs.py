"""Tests for finance and pricing environments."""

from __future__ import annotations

import importlib.util
import unittest

from agents.q_learning_agent import QLearningAgent
from environments.cashflow_env import CashFlowConfig, CashFlowEnv
from environments.pricing_env import PricingConfig, PricingEnv


class CashFlowEnvTests(unittest.TestCase):
    def test_cashflow_reset_and_step(self) -> None:
        env = CashFlowEnv(CashFlowConfig(seed=1))

        state, info = env.reset(seed=1)
        next_state, reward, done, step_info = env.step("pay_debt")

        self.assertEqual(len(state), 4)
        self.assertEqual(len(next_state), 4)
        self.assertFalse(done)
        self.assertIn("net_worth", step_info)
        self.assertIsInstance(reward, float)

    def test_cashflow_rejects_invalid_action(self) -> None:
        env = CashFlowEnv(CashFlowConfig(seed=1))
        env.reset(seed=1)

        with self.assertRaises(ValueError):
            env.step("invalid")


class PricingEnvTests(unittest.TestCase):
    def test_pricing_reset_and_step(self) -> None:
        env = PricingEnv(PricingConfig(seed=1))

        state, _ = env.reset(seed=1)
        next_state, reward, done, info = env.step(28)

        self.assertEqual(len(state), 4)
        self.assertEqual(len(next_state), 4)
        self.assertFalse(done)
        self.assertIn("revenue", info)
        self.assertIsInstance(reward, float)

    def test_q_learning_supports_string_actions(self) -> None:
        agent = QLearningAgent(actions=("pay_debt", "hold_cash"), seed=1)

        agent.update((1, 2, 3, 4), "pay_debt", 10.0, (1, 2, 3, 5), True)

        self.assertEqual(agent.get_q_value((1, 2, 3, 4), "pay_debt"), 1.5)


class DQNAgentTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_dqn_choose_action(self) -> None:
        from agents.dqn_agent import DQNAgent

        agent = DQNAgent(state_size=4, actions=(0, 10), batch_size=2, seed=1)

        self.assertIn(agent.choose_action((1, 2, 3, 4), explore=False), (0, 10))


if __name__ == "__main__":
    unittest.main()

