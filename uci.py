"""Minimal UCI engine: `python uci.py runs/main/latest.pt` (add it to Cute Chess / Arena as a command)."""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch

import azchess_rs
from azchess.inference import load_evaluator, run_search
from azchess.model import default_device


def send(line: str) -> None:
    print(line, flush=True)


def parse_position(tokens: list[str]) -> tuple[str | None, list[str]]:
    if tokens and tokens[0] == "fen":
        fen, rest = " ".join(tokens[1:7]), tokens[7:]
    else:
        fen, rest = None, tokens[1:]
    return fen, rest[1:] if rest and rest[0] == "moves" else []


def search_budget(tokens: list[str], white_to_move: bool, default_nodes: int) -> tuple[int, float | None]:
    args = {k: int(v) for k, v in zip(tokens, tokens[1:]) if v.lstrip("-").isdigit()}
    if "nodes" in args:
        return args["nodes"], None
    time_key, inc_key = ("wtime", "winc") if white_to_move else ("btime", "binc")
    if "movetime" in args:
        seconds = args["movetime"] / 1000
    elif time_key in args:
        left = args[time_key] / 1000
        seconds = min(left / args.get("movestogo", 30) + args.get(inc_key, 0) / 1000 * 0.8, left * 0.5)
    else:
        return default_nodes, None
    return 10**9, time.monotonic() + max(seconds - 0.05, 0.01)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--nodes", type=int, default=800)
    p.add_argument("--batch", type=int, default=16, help="leaves per network call (virtual loss)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device) if args.device else default_device()
    evaluator = load_evaluator(args.checkpoint, device)
    searcher = azchess_rs.Searcher(num_slots=1, leaves_per_step=args.batch)
    nodes, fen, moves = args.nodes, None, []

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
            fen, moves = None, []
        elif cmd == "position":
            fen, moves = parse_position(tokens[1:])
        elif cmd == "go":
            searcher.set_position(0, fen, moves)
            if not azchess_rs.legal_moves(fen, moves):
                send("bestmove 0000")
                continue
            white = (fen is None or fen.split()[1] == "w") == (len(moves) % 2 == 0)
            n, deadline = search_budget(tokens[1:], white, nodes)
            t0 = time.monotonic()
            run_search(searcher, evaluator, n, deadline=deadline, clock=time.monotonic)
            best, q, visits, pv = searcher.result(0)
            cp = round(111.714640912 * math.tan(1.5620688421 * max(-0.999, min(0.999, q))))
            nps = int(visits / max(time.monotonic() - t0, 1e-6))
            send(f"info nodes {visits} nps {nps} score cp {cp} pv {' '.join(pv)}")
            send(f"bestmove {best}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
