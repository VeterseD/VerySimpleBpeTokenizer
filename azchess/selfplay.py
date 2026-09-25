"""Self-play: many games in parallel, one batched network call per simulation step."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from .encoding import encode_board
from .mcts import SearchConfig, Tree, run_search, terminal_value


@dataclass
class SelfPlayConfig:
    simulations: int = 200
    parallel_games: int = 128
    temperature_plies: int = 30
    max_plies: int = 400
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass
class GameRecord:
    planes: list = field(default_factory=list)
    policy_idx: list = field(default_factory=list)
    policy_p: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    result: float = 0.0  # white's perspective

    @property
    def plies(self) -> int:
        return len(self.turns)

    def value_targets(self) -> np.ndarray:
        return np.array([self.result if t == chess.WHITE else -self.result for t in self.turns], dtype=np.float32)


def game_result(board: chess.Board, plies: int, max_plies: int) -> float | None:
    """Result from white's perspective, or None if the game continues. Length cap counts as a draw."""
    value = terminal_value(board, list(board.legal_moves))
    if value is not None:
        return value if board.turn == chess.WHITE else -value
    return 0.0 if plies >= max_plies else None


def play_games(evaluator, n_games: int, cfg: SelfPlayConfig, rng: np.random.Generator) -> tuple[list[GameRecord], int]:
    finished: list[GameRecord] = []
    active: list[tuple[chess.Board, Tree, GameRecord]] = []
    started = total_sims = 0
    while len(finished) < n_games:
        while len(active) < cfg.parallel_games and started < n_games:
            active.append((chess.Board(), Tree(noise_rng=rng), GameRecord()))
            started += 1

        total_sims += run_search([a[0] for a in active], [a[1] for a in active], evaluator, cfg.simulations, cfg.search)

        still_playing = []
        for board, tree, rec in active:
            root = tree.root
            visits = root.N.sum()
            probs = root.N / visits if visits > 0 else root.P
            rec.planes.append(encode_board(board))
            rec.policy_idx.append(root.idx)
            rec.policy_p.append(probs.astype(np.float32))
            rec.turns.append(board.turn)

            if rec.plies <= cfg.temperature_plies:
                i = int(rng.choice(len(probs), p=probs))
            else:
                i = tree.best_index()
            board.push(root.moves[i])
            tree.advance(i)

            result = game_result(board, rec.plies, cfg.max_plies)
            if result is None:
                still_playing.append((board, tree, rec))
            else:
                rec.result = result
                finished.append(rec)
        active = still_playing
    return finished, total_sims
