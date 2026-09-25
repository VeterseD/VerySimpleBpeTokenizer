"""Fixed-size ring buffer of training positions with sparse policy targets."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .encoding import MAX_LEGAL_MOVES, NUM_PLANES


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.planes = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.policy_idx = np.full((capacity, MAX_LEGAL_MOVES), -1, dtype=np.int16)
        self.policy_p = np.zeros((capacity, MAX_LEGAL_MOVES), dtype=np.float16)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add_batch(self, planes: np.ndarray, idx: np.ndarray, probs: np.ndarray, value: np.ndarray) -> None:
        """idx/probs are padded to MAX_LEGAL_MOVES with -1 / 0."""
        n = len(value)
        if n > self.capacity:
            planes, idx, probs, value = (a[-self.capacity :] for a in (planes, idx, probs, value))
            n = self.capacity
        rows = (self.pos + np.arange(n)) % self.capacity
        self.planes[rows] = planes
        self.policy_idx[rows] = idx
        self.policy_p[rows] = probs
        self.value[rows] = value
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

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
