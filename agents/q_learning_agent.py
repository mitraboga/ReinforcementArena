"""Tabular Q-learning agent."""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Hashable, Iterable


class QLearningAgent:
    """A tabular Q-learning agent for discrete state and action spaces."""

    def __init__(
        self,
        actions: Iterable[Hashable],
        alpha: float = 0.15,
        gamma: float = 0.92,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.03,
        seed: int | None = None,
    ):
        self.actions = tuple(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q: dict[tuple[tuple[int, ...], Hashable], float] = {}
        self.rng = random.Random(seed)

    def get_q_value(self, state: tuple[int, ...], action: Hashable) -> float:
        """Return the learned Q-value for a state/action pair."""

        return self.q.get((tuple(state), action), 0.0)

    def choose_action(self, state: tuple[int, ...], explore: bool = True) -> Hashable:
        """Choose an action using an epsilon-greedy policy."""

        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.best_action(state)

    def best_action(self, state: tuple[int, ...]) -> Hashable:
        """Return an action with the highest estimated Q-value."""

        best_value = max(self.get_q_value(state, action) for action in self.actions)
        best_actions = [
            action for action in self.actions if self.get_q_value(state, action) == best_value
        ]
        return self.rng.choice(best_actions)

    def update(
        self,
        state: tuple[int, ...],
        action: Hashable,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        """Apply the Q-learning update rule."""

        old_q = self.get_q_value(state, action)
        future_reward = 0.0 if done else self.best_future_reward(next_state)
        new_estimate = reward + self.gamma * future_reward
        self.q[(tuple(state), action)] = old_q + self.alpha * (new_estimate - old_q)

    def best_future_reward(self, state: tuple[int, ...]) -> float:
        """Return the best known future reward from a state."""

        if not self.actions:
            return 0.0
        return max(self.get_q_value(state, action) for action in self.actions)

    def decay_epsilon(self) -> None:
        """Reduce exploration after an episode."""

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Persist the learned policy and hyperparameters."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """Load a persisted Q-learning agent."""

        with Path(path).open("rb") as file:
            agent = pickle.load(file)
        if not isinstance(agent, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(agent).__name__}")
        return agent
