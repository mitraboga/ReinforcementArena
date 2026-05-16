"""Deep Q-Network agent for discrete-action environments."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np


class Experience(NamedTuple):
    state: tuple[int, ...]
    action_index: int
    reward: float
    next_state: tuple[int, ...]
    done: bool


class DQNAgent:
    """PyTorch DQN agent with replay memory and a target network."""

    def __init__(
        self,
        state_size: int,
        actions: Iterable[int],
        hidden_size: int = 128,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        learning_rate: float = 0.001,
        replay_size: int = 10000,
        batch_size: int = 64,
        target_update_frequency: int = 100,
        seed: int | None = None,
    ):
        try:
            import torch
            from torch import nn, optim
        except ImportError as exc:
            raise ImportError(
                "DQNAgent requires PyTorch. Install project dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.optim = optim
        self.actions = tuple(actions)
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.memory: deque[Experience] = deque(maxlen=replay_size)
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.policy_net = self._build_network(hidden_size)
        self.target_net = self._build_network(hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.training_steps = 0

    def _build_network(self, hidden_size: int):
        return self.nn.Sequential(
            self.nn.Linear(self.state_size, hidden_size),
            self.nn.ReLU(),
            self.nn.Linear(hidden_size, hidden_size),
            self.nn.ReLU(),
            self.nn.Linear(hidden_size, len(self.actions)),
        )

    def choose_action(self, state: tuple[int, ...], explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        with self.torch.no_grad():
            q_values = self.policy_net(self._state_tensor(state).unsqueeze(0))
        return self.actions[int(q_values.argmax().item())]

    def remember(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        self.memory.append(
            Experience(state, self.actions.index(action), reward, next_state, done)
        )

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.rng.sample(list(self.memory), self.batch_size)
        states = self.torch.stack([self._state_tensor(item.state) for item in batch])
        actions = self.torch.tensor([item.action_index for item in batch], dtype=self.torch.long)
        rewards = self.torch.tensor([item.reward for item in batch], dtype=self.torch.float32)
        next_states = self.torch.stack([self._state_tensor(item.next_state) for item in batch])
        dones = self.torch.tensor([item.done for item in batch], dtype=self.torch.float32)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with self.torch.no_grad():
            next_q = self.target_net(next_states).max(1).values
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_size": self.state_size,
                "actions": self.actions,
                "epsilon": self.epsilon,
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "DQNAgent":
        """Load a DQN checkpoint.

        Network hyperparameters such as ``hidden_size`` must match training
        config when they differ from defaults.
        """

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "DQNAgent requires PyTorch. Install project dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        checkpoint = torch.load(path, map_location="cpu")
        agent = cls(
            state_size=checkpoint["state_size"],
            actions=checkpoint["actions"],
            epsilon=checkpoint.get("epsilon", 0.0),
            **kwargs,
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.policy_net.eval()
        agent.target_net.eval()
        return agent

    def _state_tensor(self, state: tuple[int, ...]):
        return self.torch.tensor(state, dtype=self.torch.float32)
