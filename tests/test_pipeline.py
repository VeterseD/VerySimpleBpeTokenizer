import subprocess
import sys
from pathlib import Path

import chess
import numpy as np

import arena
import train
from azchess.replay import ReplayBuffer
from azchess.selfplay import GameRecord

ROOT = Path(__file__).resolve().parent.parent


def test_replay_buffer_roundtrip_keeps_newest(tmp_path):
    buf = ReplayBuffer(4)
    rec = GameRecord()
    for k in range(6):
        rec.planes.append(np.full((20, 8, 8), k, np.uint8))
        rec.policy_idx.append(np.array([k, k + 1]))
        rec.policy_p.append(np.array([0.25, 0.75], np.float32))
        rec.turns.append(chess.WHITE if k % 2 == 0 else chess.BLACK)
    rec.result = 1.0
    buf.add_game(rec)
    buf.save(tmp_path / "b.npz")

    small = ReplayBuffer(3)
    small.load(tmp_path / "b.npz")
    assert small.size == 3
    assert [int(small.planes[i, 0, 0, 0]) for i in range(3)] == [3, 4, 5]
    assert list(small.value) == [-1.0, 1.0, -1.0]


def test_train_resume_uci_and_arena(tmp_path, capsys):
    common = [
        "--run-dir", str(tmp_path), "--blocks", "1", "--channels", "16", "--games-per-iter", "3",
        "--parallel-games", "2", "--simulations", "8", "--max-plies", "24", "--buffer-size", "1000",
        "--min-buffer", "10", "--batch-size", "16", "--device", "cpu", "--no-amp", "--save-every", "1",
    ]
    train.main(common + ["--iterations", "1"])
    train.main(common + ["--iterations", "2"])
    out = capsys.readouterr().out
    assert "resumed" in out and "[2] train:" in out
    ckpt = tmp_path / "latest.pt"
    assert ckpt.exists() and (tmp_path / "iter_0002.pt").exists()

    proc = subprocess.run(
        [sys.executable, str(ROOT / "uci.py"), str(ckpt), "--device", "cpu"],
        input="uci\nisready\nposition startpos moves e2e4\ngo nodes 32\ngo movetime 200\nquit\n",
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    best = [line.split()[1] for line in proc.stdout.splitlines() if line.startswith("bestmove")]
    board = chess.Board()
    board.push_uci("e2e4")
    assert len(best) == 2 and all(chess.Move.from_uci(m) in board.legal_moves for m in best)

    arena.main([str(ckpt), "random", "--games", "4", "--simulations", "8", "--max-plies", "30", "--device", "cpu"])
    assert "A vs B: +" in capsys.readouterr().out
