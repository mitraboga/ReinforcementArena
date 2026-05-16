"""Tabular SARSA agent."""

from __future__ import annotations

from typing import Iterable

from agents.q_learning_agent import QLearningAgent


class SARSAAgent(QLearningAgent):
    """On-policy temporal-difference learning agent."""

    def __init__(
        self,
        actions: Iterable[int],
        alpha: float = 0.15,
        gamma: float = 0.92,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.03,
        seed: int | None = None,
    ):
        super().__init__(
            actions=actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            seed=seed,
        )

    def update_sarsa(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        next_action: int | None,
        done: bool,
    ) -> None:
        """Apply the SARSA update rule using the next policy action."""

        old_q = self.get_q_value(state, action)
        next_q = 0.0 if done or next_action is None else self.get_q_value(next_state, next_action)
        new_estimate = reward + self.gamma * next_q
        self.q[(tuple(state), action)] = old_q + self.alpha * (new_estimate - old_q)
