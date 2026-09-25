"""Self-play games as PGN files, with MCTS info as move comments."""
from __future__ import annotations

import datetime
from pathlib import Path

# PGN Termination header -> short label for the training log
TERMINATION_SHORT = {
    "checkmate": "mate",
    "stalemate": "stalemate",
    "threefold repetition": "rep",
    "fifty-move rule": "50move",
    "insufficient material": "material",
    "max plies": "maxplies",
}


def result_str(result: float) -> str:
    return "1-0" if result > 0 else "0-1" if result < 0 else "1/2-1/2"


def pgn_game(movetext: str, headers: dict) -> str:
    lines = [f'[{k} "{v}"]' for k, v in headers.items()]
    return "\n".join(lines) + "\n\n" + movetext + "\n\n"


class GameLogger:
    """Appends games to <dir>/selfplay_<first game number>.pgn, starting a new file every `per_file` games."""

    def __init__(self, directory: Path, every: int = 1, per_file: int = 10_000, first_game: int = 0):
        self.directory = Path(directory)
        self.every = every
        self.per_file = per_file
        self.game_no = first_game
        self.file = None
        self.file_start = None

    def _open_for(self, game_no: int):
        start = game_no - game_no % self.per_file
        if self.file is None or start != self.file_start:
            if self.file is not None:
                self.file.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            self.file = open(self.directory / f"selfplay_{start:09d}.pgn", "a", encoding="utf-8")
            self.file_start = start
        return self.file

    def write(self, movetexts, results, terminations, plies, net_step: int, worker: int) -> None:
        date = datetime.date.today().strftime("%Y.%m.%d")
        wrote = False
        for movetext, result, termination, n in zip(movetexts, results, terminations, plies):
            self.game_no += 1
            if self.every <= 0 or self.game_no % self.every:
                continue
            headers = {
                "Event": "azchess self-play",
                "Site": f"worker {worker}",
                "Date": date,
                "Round": self.game_no,
                "White": f"azchess step {net_step}",
                "Black": f"azchess step {net_step}",
                "Result": result_str(result),
                "Termination": termination,
                "PlyCount": int(n),
                "NetStep": net_step,
            }
            self._open_for(self.game_no).write(pgn_game(movetext, headers))
            wrote = True
        if wrote:
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
