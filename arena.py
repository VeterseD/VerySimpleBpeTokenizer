"""Match two checkpoints (or a checkpoint vs random) and estimate the Elo difference."""
from __future__ import annotations

import argparse
import math

import chess
import numpy as np
import torch

from azchess.mcts import Evaluator, SearchConfig, Tree, run_search
from azchess.model import amp_dtype_for, default_device, load_checkpoint
from azchess.selfplay import game_result


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
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--parallel", type=int, default=64)
    p.add_argument("--opening-plies", type=int, default=4, help="random opening shared by each color-swapped pair")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device(args.device) if args.device else default_device()
    amp = amp_dtype_for(device)
    rng = np.random.default_rng(args.seed)
    cfg = SearchConfig()
    eval_a = Evaluator(load_checkpoint(args.model_a, device)[0], device, amp)
    eval_b = None if args.model_b == "random" else Evaluator(load_checkpoint(args.model_b, device)[0], device, amp)

    queue = []
    for _ in range((args.games + 1) // 2):
        opening = random_opening(rng, args.opening_plies)
        queue += [(opening, True), (opening, False)]
    queue = queue[: args.games]

    score = {"w": 0, "d": 0, "l": 0}
    active = []
    while queue or active:
        while queue and len(active) < args.parallel:
            opening, a_white = queue.pop()
            board = chess.Board()
            for m in opening:
                board.push(m)
            active.append([board, a_white, 0])

        for evaluator, a_moves in ((eval_a, True), (eval_b, False)):
            group = [g for g in active if (g[0].turn == chess.WHITE) == (g[1] == a_moves)]
            if not group:
                continue
            if evaluator is None:
                for g in group:
                    legal = list(g[0].legal_moves)
                    g[0].push(legal[rng.integers(len(legal))])
                continue
            trees = [Tree() for _ in group]
            run_search([g[0] for g in group], trees, evaluator, args.simulations, cfg)
            for g, tree in zip(group, trees):
                g[0].push(tree.root.moves[tree.best_index()])

        still = []
        for g in active:
            g[2] += 1
            result = game_result(g[0], g[2], args.max_plies)
            if result is None:
                still.append(g)
                continue
            a_result = result if g[1] else -result
            score["w" if a_result > 0 else "l" if a_result < 0 else "d"] += 1
        active = still

    n = sum(score.values())
    s = (score["w"] + 0.5 * score["d"]) / n
    elo = -400 * math.log10(1 / s - 1) if 0 < s < 1 else math.copysign(math.inf, s - 0.5)
    print(f"A vs B: +{score['w']} ={score['d']} -{score['l']}  score {s:.3f}  Elo diff {elo:+.0f}")


if __name__ == "__main__":
    main()
