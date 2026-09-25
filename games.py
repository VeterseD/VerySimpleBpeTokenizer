"""Browse logged games (self-play or arena PGN).

  python games.py runs/main/games                        summary of all games
  python games.py runs/main/games --result 1-0 --list 20 last 20 white wins
  python games.py runs/main/games --termination checkmate --min-plies 100 --list 10
  python games.py runs/main/games --show 12345           one game move by move with its search info

Comments on fully searched moves: N = root visits, Q = root value for the side to move (-1..1),
then the top 3 branches: move, visits, prior p from the network, q = search value of that move.
"sampled" means the move was drawn by temperature instead of being the most visited one.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import chess
import chess.pgn


def pgn_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.pgn")) if path.is_dir() else [path]


def iter_headers(files):
    for f in files:
        with open(f, encoding="utf-8") as fh:
            while True:
                offset = fh.tell()
                headers = chess.pgn.read_headers(fh)
                if headers is None:
                    break
                yield f, offset, headers


def matches(h, args) -> bool:
    if args.result and h.get("Result") != args.result:
        return False
    if args.termination and args.termination not in h.get("Termination", ""):
        return False
    plies = int(h.get("PlyCount", 0))
    if args.min_plies is not None and plies < args.min_plies:
        return False
    if args.max_plies is not None and plies > args.max_plies:
        return False
    if args.min_step is not None and int(h.get("NetStep", 0)) < args.min_step:
        return False
    return True


def show(f: Path, offset: int) -> None:
    with open(f, encoding="utf-8") as fh:
        fh.seek(offset)
        game = chess.pgn.read_game(fh)
    for k, v in game.headers.items():
        print(f"{k}: {v}")
    print()
    board = game.board()
    for node in game.mainline():
        prefix = f"{board.fullmove_number}." if board.turn == chess.WHITE else f"{board.fullmove_number}..."
        san = board.san(node.move)
        board.push(node.move)
        print(f"{prefix:<6}{san:<8}{node.comment}")
    print(f"\nfinal position: {board.fen()}\n{board}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="PGN file or directory with PGN files")
    p.add_argument("--result", choices=["1-0", "0-1", "1/2-1/2"])
    p.add_argument("--termination", help="substring, e.g. checkmate, repetition, fifty, material, max")
    p.add_argument("--min-plies", type=int)
    p.add_argument("--max-plies", type=int)
    p.add_argument("--min-step", type=int, help="only games played by nets from this training step on")
    p.add_argument("--list", type=int, metavar="N", help="print the last N matching games")
    p.add_argument("--show", type=int, metavar="ROUND", help="print game with this Round number")
    args = p.parse_args(argv)

    files = pgn_files(Path(args.path))
    if args.show is not None:
        for f, offset, h in iter_headers(files):
            if h.get("Round") == str(args.show):
                show(f, offset)
                return
        raise SystemExit(f"game {args.show} not found")

    total, plies = 0, 0
    results, endings = collections.Counter(), collections.Counter()
    recent = collections.deque(maxlen=args.list or 0)
    for f, offset, h in iter_headers(files):
        if not matches(h, args):
            continue
        total += 1
        plies += int(h.get("PlyCount", 0))
        results[h.get("Result")] += 1
        endings[h.get("Termination", "?")] += 1
        recent.append(h)
    if not total:
        print("no matching games")
        return
    print(f"{total} games, avg {plies / total:.0f} plies")
    print("results:     " + "  ".join(f"{k} {100 * v / total:.1f}%" for k, v in results.most_common()))
    print("termination: " + "  ".join(f"{k} {100 * v / total:.1f}%" for k, v in endings.most_common()))
    if recent:
        print(f"\n{'round':>9} {'step':>8} {'result':>8} {'plies':>6}  termination")
        for h in recent:
            print(f"{h.get('Round'):>9} {h.get('NetStep', '-'):>8} {h.get('Result'):>8} "
                  f"{h.get('PlyCount', '?'):>6}  {h.get('Termination', '')}")


if __name__ == "__main__":
    main()
