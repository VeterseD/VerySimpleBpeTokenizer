"""Match two checkpoints (or a checkpoint vs random) and estimate the Elo difference."""
from __future__ import annotations

import argparse
import math

import chess
import numpy as np
import torch

import azchess_rs
from azchess.inference import load_evaluator, run_search
from azchess.model import default_device


def game_result(board: chess.Board, plies: int, max_plies: int) -> float | None:
    """White's result, or None if the game continues. Same draw rules as search; length cap is a draw."""
    if not any(board.legal_moves):
        return (-1.0 if board.turn == chess.WHITE else 1.0) if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    return 0.0 if plies >= max_plies else None


def random_opening(rng: np.random.Generator, plies: int) -> list[chess.Move]:
    while True:
        board, moves = chess.Board(), []
        for _ in range(plies):
            legal = list(board.legal_moves)
            move = legal[rng.integers(len(legal))]
            board.push(move)
            moves.append(move)
        if game_result(board, 0, 10**9) is None:
            return moves


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_a")
    p.add_argument("model_b", help="checkpoint path or 'random'")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--parallel", type=int, default=128)
    p.add_argument("--opening-plies", type=int, default=4, help="random opening shared by each color-swapped pair")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device(args.device) if args.device else default_device()
    rng = np.random.default_rng(args.seed)
    players = [load_evaluator(args.model_a, device)]
    players.append(None if args.model_b == "random" else load_evaluator(args.model_b, device))
    searchers = [azchess_rs.Searcher(num_slots=args.parallel, leaves_per_step=1) for _ in players]

    queue = []
    for _ in range((args.games + 1) // 2):
        opening = random_opening(rng, args.opening_plies)
        queue += [(opening, True), (opening, False)]
    queue = queue[: args.games]

    score = {"w": 0, "d": 0, "l": 0}
    slots: list = [None] * args.parallel  # (board, a_is_white, plies)
    while queue or any(slots):
        for s in range(args.parallel):
            if slots[s] is None and queue:
                opening, a_white = queue.pop()
                board = chess.Board()
                for m in opening:
                    board.push(m)
                slots[s] = [board, a_white, 0]

        for k, (evaluator, searcher) in enumerate(zip(players, searchers)):
            to_move = [s for s, g in enumerate(slots) if g and (g[0].turn == chess.WHITE) == (g[1] == (k == 0))]
            if not to_move:
                continue
            if evaluator is None:
                for s in to_move:
                    legal = list(slots[s][0].legal_moves)
                    slots[s][0].push(legal[rng.integers(len(legal))])
                continue
            for s in to_move:
                searcher.set_position(s, None, [m.uci() for m in slots[s][0].move_stack])
            run_search(searcher, evaluator, args.simulations, to_move)
            for s in to_move:
                slots[s][0].push_uci(searcher.result(s)[0])

        for s, g in enumerate(slots):
            if g is None:
                continue
            g[2] += 1
            result = game_result(g[0], g[2], args.max_plies)
            if result is not None:
                a_result = result if g[1] else -result
                score["w" if a_result > 0 else "l" if a_result < 0 else "d"] += 1
                slots[s] = None

    n = sum(score.values())
    s = (score["w"] + 0.5 * score["d"]) / n
    elo = -400 * math.log10(1 / s - 1) if 0 < s < 1 else math.copysign(math.inf, s - 0.5)
    print(f"A vs B: +{score['w']} ={score['d']} -{score['l']}  score {s:.3f}  Elo diff {elo:+.0f}")


if __name__ == "__main__":
    main()
