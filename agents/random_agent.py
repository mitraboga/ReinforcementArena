"""Random baseline policy."""

from __future__ import annotations

import random
from typing import Iterable


class RandomAgent:
    """Chooses uniformly among available reorder quantities."""

    def __init__(self, actions: Iterable[int], seed: int | None = None):
        self.actions = tuple(actions)
        self.rng = random.Random(seed)

    def choose_action(self, state: tuple[int, ...], explore: bool = False) -> int:
        return self.rng.choice(self.actions)

