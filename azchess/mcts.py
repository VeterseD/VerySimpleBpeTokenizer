"""Batched PUCT search: many trees descend together and share one network batch."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
import numpy as np
import torch

from .encoding import encode_board, move_to_index


@dataclass
class SearchConfig:
    c_puct: float = 1.5
    fpu_reduction: float = 0.25
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


class Node:
    """Edge statistics live in the parent; W is from the perspective of the player to move here."""

    __slots__ = ("moves", "idx", "P", "N", "W", "children", "terminal")

    def __init__(self):
        self.moves = None
        self.idx = None
        self.P = None
        self.N = None
        self.W = None
        self.children = None
        self.terminal = None

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def expand(self, moves: list[chess.Move], idx: list[int], logits: np.ndarray) -> None:
        self.moves = moves
        self.idx = np.asarray(idx, dtype=np.int64)
        x = logits[self.idx].astype(np.float64)
        p = np.exp(x - x.max())
        self.P = p / p.sum()
        self.N = np.zeros(len(moves))
        self.W = np.zeros(len(moves))
        self.children = [None] * len(moves)

    def select(self, cfg: SearchConfig) -> int:
        total = self.N.sum()
        fpu = self.W.sum() / total - cfg.fpu_reduction if total > 0 else 0.0
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), fpu)
        u = cfg.c_puct * self.P * math.sqrt(total + 1) / (1 + self.N)
        return int(np.argmax(q + u))


class Tree:
    def __init__(self, noise_rng: np.random.Generator | None = None):
        self.root = Node()
        self.noise_rng = noise_rng
        self._noised = False

    def advance(self, i: int) -> None:
        """Reuse the subtree of the played move."""
        child = self.root.children[i] if self.root.expanded else None
        self.root = child if child is not None else Node()
        self._noised = False

    def maybe_add_noise(self, cfg: SearchConfig) -> None:
        if self.noise_rng is None or self._noised or not self.root.expanded:
            return
        eta = self.noise_rng.dirichlet([cfg.dirichlet_alpha] * len(self.root.P))
        self.root.P = (1 - cfg.dirichlet_frac) * self.root.P + cfg.dirichlet_frac * eta
        self._noised = True

    def best_index(self) -> int:
        return int(np.argmax(self.root.N))


def terminal_value(board: chess.Board, legal: list[chess.Move]) -> float | None:
    """Game value for the side to move, or None if the game continues."""
    if not legal:
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    return None


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device, amp_dtype: torch.dtype | None = None):
        self.model = model
        self.device = device
        self.amp_dtype = amp_dtype

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(planes).to(self.device)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            logits, value = self.model(x)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()


def _backprop(path: list[tuple[Node, int]], value: float) -> None:
    # value is for the side to move at the leaf; each edge belongs to the opponent of the node below it.
    for node, i in reversed(path):
        value = -value
        node.N[i] += 1
        node.W[i] += value


def _descend(board: chess.Board, tree: Tree, cfg: SearchConfig):
    """Walks to a leaf. Terminal leaves are backed up here and return None; otherwise returns what to evaluate."""
    node, path, pushed = tree.root, [], 0
    try:
        while node.expanded and node.terminal is None:
            i = node.select(cfg)
            path.append((node, i))
            board.push(node.moves[i])
            pushed += 1
            if node.children[i] is None:
                node.children[i] = Node()
            node = node.children[i]

        if node.terminal is None:
            legal = list(board.legal_moves)
            node.terminal = terminal_value(board, legal)
            if node.terminal is None:
                idx = [move_to_index(m, board.turn) for m in legal]
                return node, path, legal, idx, encode_board(board)
    finally:
        for _ in range(pushed):
            board.pop()
    _backprop(path, node.terminal)
    return None


def run_search(
    boards: list[chess.Board],
    trees: list[Tree],
    evaluator,
    num_sims: int,
    cfg: SearchConfig,
    deadline: float | None = None,
) -> int:
    """Adds num_sims simulations to every tree (or stops at deadline). Returns total simulations done."""
    done = [0] * len(trees)
    for tree in trees:
        tree.maybe_add_noise(cfg)
    while True:
        if deadline is not None and time.monotonic() >= deadline and all(t.root.expanded for t in trees):
            break
        pending = []
        for i, (board, tree) in enumerate(zip(boards, trees)):
            while done[i] < num_sims:
                leaf = _descend(board, tree, cfg)
                done[i] += 1
                if leaf is not None:
                    pending.append((i, *leaf))
                    break
        if not pending:
            break
        logits, values = evaluator(np.stack([p[5] for p in pending]))
        for k, (i, node, path, legal, idx, _) in enumerate(pending):
            node.expand(legal, idx, logits[k])
            if node is trees[i].root:
                trees[i].maybe_add_noise(cfg)
            _backprop(path, float(values[k]))
    return sum(done)
