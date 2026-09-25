"""Fixed-size ring buffer of training positions with sparse policy targets."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .encoding import MAX_LEGAL_MOVES, NUM_PLANES
from .selfplay import GameRecord


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.planes = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.policy_idx = np.full((capacity, MAX_LEGAL_MOVES), -1, dtype=np.int16)
        self.policy_p = np.zeros((capacity, MAX_LEGAL_MOVES), dtype=np.float16)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, planes: np.ndarray, idx: np.ndarray, probs: np.ndarray, value: float) -> None:
        i, k = self.pos, len(idx)
        self.planes[i] = planes
        self.policy_idx[i] = -1
        self.policy_idx[i, :k] = idx
        self.policy_p[i] = 0
        self.policy_p[i, :k] = probs
        self.value[i] = value
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_game(self, rec: GameRecord) -> None:
        for planes, idx, probs, value in zip(rec.planes, rec.policy_idx, rec.policy_p, rec.value_targets()):
            self.add(planes, idx, probs, value)

    def sample(self, n: int, rng: np.random.Generator):
        i = rng.integers(0, self.size, n)
        return self.planes[i], self.policy_idx[i], self.policy_p[i], self.value[i]

    def save(self, path: Path) -> None:
        order = np.arange(self.pos - self.size, self.pos) % self.capacity  # oldest first
        tmp = Path(path).with_suffix(".tmp.npz")
        np.savez(
            tmp,
            planes=self.planes[order],
            policy_idx=self.policy_idx[order],
            policy_p=self.policy_p[order],
            value=self.value[order],
        )
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        with np.load(path) as data:
            n = min(len(data["value"]), self.capacity)
            self.planes[:n] = data["planes"][-n:]
            self.policy_idx[:n] = data["policy_idx"][-n:]
            self.policy_p[:n] = data["policy_p"][-n:]
            self.value[:n] = data["value"][-n:]
        self.size = n
        self.pos = n % self.capacity
