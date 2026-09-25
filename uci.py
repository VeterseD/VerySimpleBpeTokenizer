"""Minimal UCI engine: `python uci.py runs/main/latest.pt` (add it to Cute Chess / Arena as a command)."""
from __future__ import annotations

import argparse
import math
import sys
import time

import chess
import torch

from azchess.mcts import Evaluator, SearchConfig, Tree, run_search
from azchess.model import amp_dtype_for, default_device, load_checkpoint


def send(line: str) -> None:
    print(line, flush=True)


def parse_position(tokens: list[str]) -> chess.Board:
    if tokens and tokens[0] == "fen":
        board, rest = chess.Board(" ".join(tokens[1:7])), tokens[7:]
    else:
        board, rest = chess.Board(), tokens[1:]
    if rest and rest[0] == "moves":
        for uci_move in rest[1:]:
            board.push_uci(uci_move)
    return board


def search_budget(tokens: list[str], turn: chess.Color, default_nodes: int) -> tuple[int, float | None]:
    args = {k: int(v) for k, v in zip(tokens, tokens[1:]) if v.lstrip("-").isdigit()}
    if "nodes" in args:
        return args["nodes"], None
    if "movetime" in args:
        seconds = args["movetime"] / 1000
    elif ("wtime" if turn == chess.WHITE else "btime") in args:
        left = args["wtime" if turn == chess.WHITE else "btime"] / 1000
        inc = args.get("winc" if turn == chess.WHITE else "binc", 0) / 1000
        seconds = min(left / args.get("movestogo", 30) + inc * 0.8, left * 0.5)
    else:
        return default_nodes, None
    return 10**9, time.monotonic() + max(seconds - 0.05, 0.01)


def principal_variation(tree: Tree, limit: int = 10) -> list[chess.Move]:
    pv, node = [], tree.root
    while node is not None and node.expanded and node.N.sum() > 0 and len(pv) < limit:
        i = int(node.N.argmax())
        pv.append(node.moves[i])
        node = node.children[i]
    return pv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--nodes", type=int, default=800)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device) if args.device else default_device()
    evaluator = Evaluator(load_checkpoint(args.checkpoint, device)[0], device, amp_dtype_for(device))
    cfg, nodes, board = SearchConfig(), args.nodes, chess.Board()

    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            send("id name azchess")
            send("id author VeterseD")
            send(f"option name Nodes type spin default {args.nodes} min 1 max 100000000")
            send("uciok")
        elif cmd == "isready":
            send("readyok")
        elif cmd == "setoption" and len(tokens) >= 5 and tokens[2].lower() == "nodes":
            nodes = int(tokens[4])
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            board = parse_position(tokens[1:])
        elif cmd == "go":
            if not any(board.legal_moves):
                send("bestmove 0000")
                continue
            n, deadline = search_budget(tokens[1:], board.turn, nodes)
            tree, t0 = Tree(), time.monotonic()
            sims = run_search([board], [tree], evaluator, n, cfg, deadline)
            best = tree.best_index()
            q = tree.root.W[best] / max(tree.root.N[best], 1)
            cp = round(111.714640912 * math.tan(1.5620688421 * max(-0.999, min(0.999, q))))
            nps = int(sims / max(time.monotonic() - t0, 1e-6))
            pv = " ".join(m.uci() for m in principal_variation(tree))
            send(f"info nodes {sims} nps {nps} score cp {cp} pv {pv}")
            send(f"bestmove {tree.root.moves[best].uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
